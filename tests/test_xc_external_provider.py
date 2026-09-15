"""Full-grid external-XC host contract and periodic AO projection tests."""

from __future__ import annotations

import itertools
import tomllib

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core


_NAMES = itertools.count()


class _QuadraticNonlocalProvider:
    """Analytic all-point functional E = 1/2 (int rho)^2."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)
        self.calls: list[dict[str, object]] = []

    def __call__(self, features: dict[str, object]) -> dict[str, object]:
        self.calls.append(features)
        weights = np.asarray(features["grid_weights"], dtype=float)
        rho_a = np.asarray(features["rho_alpha"], dtype=float)
        rho_b = np.asarray(features["rho_beta"], dtype=float)
        integral = float(weights @ (rho_a + rho_b))
        q = self.scale * integral * weights
        zeros_grad = np.zeros((weights.size, 3))
        zeros = np.zeros(weights.size)
        return {
            "energy": 0.5 * self.scale * integral * integral,
            "v_rho_alpha": q,
            "v_rho_beta": q,
            "v_grad_alpha": zeros_grad,
            "v_grad_beta": zeros_grad,
            "v_tau_alpha": zeros,
            "v_tau_beta": zeros,
        }


class _AllFeatureProvider:
    """Differentiable test energy using rho, Cartesian grad, tau, and spins."""

    def __call__(self, features: dict[str, object]) -> dict[str, object]:
        weights = np.asarray(features["grid_weights"], dtype=float)
        rho_a = np.asarray(features["rho_alpha"], dtype=float)
        rho_b = np.asarray(features["rho_beta"], dtype=float)
        grad_a = np.asarray(features["grad_alpha"], dtype=float)
        grad_b = np.asarray(features["grad_beta"], dtype=float)
        tau_a = np.asarray(features["tau_alpha"], dtype=float)
        tau_b = np.asarray(features["tau_beta"], dtype=float)

        density_integral = float(weights @ (rho_a + 1.7 * rho_b))
        grad_integrand = (
            0.3 * np.sum(grad_a * grad_a, axis=1)
            + 0.2 * np.sum(grad_a * grad_b, axis=1)
            + 0.4 * np.sum(grad_b * grad_b, axis=1)
        )
        tau_integrand = (
            0.25 * tau_a * tau_a
            + 0.1 * tau_a * tau_b
            + 0.35 * tau_b * tau_b
        )
        return {
            "energy": (
                0.5 * density_integral * density_integral
                + float(weights @ (grad_integrand + tau_integrand))
            ),
            "v_rho_alpha": density_integral * weights,
            "v_rho_beta": 1.7 * density_integral * weights,
            "v_grad_alpha": weights[:, None] * (0.6 * grad_a + 0.2 * grad_b),
            "v_grad_beta": weights[:, None] * (0.2 * grad_a + 0.8 * grad_b),
            "v_tau_alpha": weights * (0.5 * tau_a + 0.1 * tau_b),
            "v_tau_beta": weights * (0.1 * tau_a + 0.7 * tau_b),
        }


def _external_name(
    provider: object,
    *,
    hf_exchange_fraction: float = 0.0,
    capability_version: int = 1,
    required_grid_profile: str = "",
) -> str:
    name = f"test-full-grid-xc-{next(_NAMES)}"
    vq.define_external_functional(
        name,
        provider,
        hf_exchange_fraction=hf_exchange_fraction,
        capability_version=capability_version,
        required_grid_profile=required_grid_profile,
    )
    return name


def _h2_periodic_fixture(
    *, lattice_length: float = 20.0, lattice_cutoff: float = 6.0
):
    lattice = np.eye(3) * lattice_length
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    grid_options = vq.GridOptions()
    grid_options.n_radial = 4
    grid_options.n_theta = 3
    grid_options.n_phi = 4
    grid = vq.build_periodic_becke_grid(
        system,
        grid_options=grid_options,
        image_radius_bohr=lattice_cutoff,
    )

    lattice_options = vq.LatticeSumOptions()
    lattice_options.cutoff_bohr = lattice_cutoff
    lattice_options.nuclear_cutoff_bohr = lattice_cutoff
    lattice_options.becke_image_radius_bohr = lattice_cutoff
    template = vq.compute_overlap_lattice(basis, system, lattice_options)
    home = next(
        index
        for index, cell in enumerate(template.cells)
        if np.all(np.asarray(cell.index, dtype=int) == 0)
    )

    def density(matrix: np.ndarray | dict[int, np.ndarray]):
        result = vq.compute_overlap_lattice(basis, system, lattice_options)
        for index in range(len(result.blocks)):
            result.set_block(
                index,
                np.zeros_like(np.asarray(result.blocks[index])),
            )
        blocks = {home: matrix} if not isinstance(matrix, dict) else matrix
        for index, block in blocks.items():
            result.set_block(index, np.asarray(block, dtype=float))
        return result

    return system, basis, grid, lattice_options, home, density


def test_external_registration_rejects_duplicates_and_builtin_shadowing():
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    assert not core._dynamic_alias_available(name)
    with pytest.raises(ValueError, match="already registered"):
        core.define_external_functional(name.upper(), provider)
    with pytest.raises(ValueError, match="already resolves"):
        core.define_external_functional("pbe", provider)
    with pytest.raises(ValueError, match="already registered as an external"):
        vq.define_functional(name, [("LDA_X", 1.0)])

    alias = f"test-libxc-alias-{next(_NAMES)}"
    vq.define_functional(alias, [("LDA_X", 1.0)])
    assert core._dynamic_alias_available(alias)


def test_external_family_registration_is_atomic_on_second_name_collision():
    occupied = _external_name(_QuadraticNonlocalProvider())
    first = f"test-full-grid-xc-family-{next(_NAMES)}"
    third = f"test-full-grid-xc-family-{next(_NAMES)}"

    with pytest.raises(ValueError, match="already registered"):
        core._define_external_functional_family(
            [first, occupied, third],
            _QuadraticNonlocalProvider(),
        )

    assert core._external_functional_registration(first) is None
    assert core._external_functional_registration(third) is None


def test_runtime_external_registration_does_not_falsely_cite_libxc():
    from vibeqc.output.citations import load_default_database

    name = _external_name(_QuadraticNonlocalProvider())
    assembled = load_default_database().assemble(
        method="rks", basis="sto-3g", functional=name
    )

    assert "libxc_2018" not in {citation.key for citation in assembled.citations}
    assert any(
        f"no citation route for functional {name!r}" in warning
        for warning in assembled.warnings
    )


def test_external_required_grid_capability_is_queryable_and_fails_closed():
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider, required_grid_profile="pyscf-level3")
    functional = vq.Functional(name, 1)
    assert functional.external_capabilities == {
        "version": 1,
        "required_grid_profile": "pyscf-level3",
    }
    assert vq.Functional("pbe").external_capabilities is None

    system, basis, generic_grid, options, _, make_density = _h2_periodic_fixture()
    assert generic_grid.atomic_grid_profile == vq.AtomicGridProfile.Generic
    density = make_density(0.4 * np.eye(basis.nbasis))
    with pytest.raises(ValueError, match="requires grid profile 'pyscf-level3'"):
        vq.build_xc_periodic(
            basis,
            system,
            generic_grid,
            functional,
            density,
            options,
        )
    assert provider.calls == []

    exact_options = vq.GridOptions()
    exact_options.atomic_grid_profile = "pyscf-level3"
    exact_grid = vq.build_periodic_becke_grid(
        system,
        grid_options=exact_options,
        image_radius_bohr=options.becke_image_radius_bohr,
    )
    assert exact_grid.atomic_grid_profile == vq.AtomicGridProfile.PySCFLevel3
    vq.build_xc_periodic(
        basis,
        system,
        exact_grid,
        functional,
        density,
        options,
    )
    assert provider.calls[-1]["grid_profile"] == "pyscf-level3"


def test_external_registration_rejects_unknown_capability_contracts():
    provider = _QuadraticNonlocalProvider()
    with pytest.raises(ValueError, match="unsupported capability version"):
        _external_name(provider, capability_version=2)
    with pytest.raises(ValueError, match="unsupported required_grid_profile"):
        _external_name(provider, required_grid_profile="private-grid")


def test_external_pointwise_and_second_derivative_apis_fail_closed():
    name = _external_name(_QuadraticNonlocalProvider())
    func = vq.Functional(name, 1)
    values = np.ones(2)

    with pytest.raises(RuntimeError, match="full-grid evaluation"):
        func.eval_unpolarised(values, values)
    with pytest.raises(RuntimeError, match="second derivatives"):
        func.eval_unpolarised_fxc(values, values)


def test_external_periodic_home_only_projection_matches_finite_difference():
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis, grid, options, home, make_density = _h2_periodic_fixture()
    func = vq.Functional(name, 1)
    assert func.is_external
    assert func.kind == vq.XCKind.MGGA

    p0 = 0.4 * np.eye(basis.nbasis)
    result = vq.build_xc_periodic(
        basis, system, grid, func, make_density(p0), options
    )
    step = 1.0e-6
    plus = p0.copy()
    minus = p0.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    e_plus = vq.build_xc_periodic(
        basis, system, grid, func, make_density(plus), options
    ).e_xc
    e_minus = vq.build_xc_periodic(
        basis, system, grid, func, make_density(minus), options
    ).e_xc
    finite_difference = (e_plus - e_minus) / (2.0 * step)

    assert np.asarray(result.V_xc.blocks[home])[0, 0] == pytest.approx(
        finite_difference, rel=2.0e-7, abs=2.0e-8
    )
    payload = provider.calls[0]
    assert payload["grid_profile"] == "generic"
    assert payload["periodic"] is True
    assert payload["periodic_dimension"] == 3
    assert np.asarray(payload["lattice"]).shape == (3, 3)
    sizes = list(payload["atomic_grid_sizes"])
    assert len(sizes) == 2
    assert sum(sizes) == len(np.asarray(grid.weights))
    assert np.asarray(payload["atomic_grid_weights"]).shape == np.asarray(
        grid.weights
    ).shape


def test_periodic_density_domain_overrides_zero_cross_cell_heuristic():
    from vibeqc.periodic_external_xc import (
        periodic_xc_difference_closed_cells,
    )

    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis, grid, options, home, make_density = _h2_periodic_fixture(
        lattice_length=4.0,
        lattice_cutoff=4.1,
    )
    func = vq.Functional(name, 1)
    p0 = 0.3 * np.eye(basis.nbasis)

    periodic_cells = periodic_xc_difference_closed_cells(
        system,
        options.cutoff_bohr,
        options.becke_image_radius_bohr,
    )
    periodic_home = next(
        index
        for index, cell in enumerate(periodic_cells)
        if np.all(np.asarray(cell.index, dtype=int) == 0)
    )

    def periodic_density(matrix: np.ndarray):
        blocks = [
            np.zeros((basis.nbasis, basis.nbasis), dtype=float)
            for _ in periodic_cells
        ]
        blocks[periodic_home] = np.asarray(matrix, dtype=float)
        return core.make_lattice_matrix_set(
            basis.nbasis,
            list(periodic_cells),
            blocks,
        )

    automatic = vq.build_xc_periodic(
        basis,
        system,
        grid,
        func,
        make_density(p0),
        options,
    )
    molecular = vq.build_xc_periodic(
        basis,
        system,
        grid,
        func,
        make_density(p0),
        options,
        density_domain=vq.PeriodicXCDensityDomain.MOLECULAR_HOME,
    )
    periodic = vq.build_xc_periodic(
        basis,
        system,
        grid,
        func,
        periodic_density(p0),
        options,
        density_domain=vq.PeriodicXCDensityDomain.PERIODIC_LATTICE,
    )

    assert automatic.e_xc == pytest.approx(molecular.e_xc, abs=1.0e-14)
    assert periodic.e_xc != pytest.approx(molecular.e_xc, rel=1.0e-5)

    step = 1.0e-6
    plus = p0.copy()
    minus = p0.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    e_plus = vq.build_xc_periodic(
        basis,
        system,
        grid,
        func,
        periodic_density(plus),
        options,
        density_domain=vq.PeriodicXCDensityDomain.PERIODIC_LATTICE,
    ).e_xc
    e_minus = vq.build_xc_periodic(
        basis,
        system,
        grid,
        func,
        periodic_density(minus),
        options,
        density_domain=vq.PeriodicXCDensityDomain.PERIODIC_LATTICE,
    ).e_xc
    finite_difference = (e_plus - e_minus) / (2.0 * step)
    assert np.asarray(periodic.V_xc.blocks[periodic_home])[0, 0] == pytest.approx(
        finite_difference,
        rel=3.0e-6,
        abs=3.0e-7,
    )


def test_explicit_periodic_density_domain_rejects_unclosed_native_input():
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis, grid, options, _, make_density = _h2_periodic_fixture(
        lattice_length=4.0,
        lattice_cutoff=4.1,
    )
    func = vq.Functional(name, 1)

    # The overlap builder's ordinary cutoff domain contains the active cells
    # but not every difference g = s-a between them.  Explicit periodic XC
    # must reject that incomplete native input instead of silently dropping
    # potential contributions at the domain boundary.
    with pytest.raises(ValueError, match="not closed over active AO-image"):
        vq.build_xc_periodic(
            basis,
            system,
            grid,
            func,
            make_density(0.3 * np.eye(basis.nbasis)),
            options,
            density_domain=vq.PeriodicXCDensityDomain.PERIODIC_LATTICE,
        )


def test_external_periodic_all_feature_projection_matches_finite_differences():
    name = _external_name(_AllFeatureProvider())
    system, basis, grid, options, home, make_density = _h2_periodic_fixture(
        lattice_length=8.0,
        lattice_cutoff=8.1,
    )
    func = vq.Functional(name, 1)
    template = make_density(np.zeros((basis.nbasis, basis.nbasis)))
    cells = [tuple(np.asarray(cell.index, dtype=int)) for cell in template.cells]
    cell_position = {cell: index for index, cell in enumerate(cells)}
    non_home = next(
        index
        for index, cell in enumerate(cells)
        if index != home and tuple(-value for value in cell) in cell_position
    )
    opposite = cell_position[tuple(-value for value in cells[non_home])]

    base_home = np.array([[0.42, 0.025], [0.025, 0.31]])
    base_cross = np.array([[0.012, -0.004], [0.007, 0.009]])
    base_blocks = {
        home: base_home,
        non_home: base_cross,
        opposite: base_cross.T,
    }
    result = vq.build_xc_periodic(
        basis, system, grid, func, make_density(base_blocks), options
    )

    # Perturb only physical real-space density variables: the home block stays
    # symmetric and every +g element is paired with its transpose at -g.
    probes = (
        ((home, 0, 0),),
        ((home, 0, 1), (home, 1, 0)),
        ((non_home, 1, 0), (opposite, 0, 1)),
    )
    step = 2.0e-7
    for probe in probes:
        plus_blocks = {index: value.copy() for index, value in base_blocks.items()}
        minus_blocks = {index: value.copy() for index, value in base_blocks.items()}
        for block_index, row, column in probe:
            plus_blocks[block_index][row, column] += step
            minus_blocks[block_index][row, column] -= step
        e_plus = vq.build_xc_periodic(
            basis, system, grid, func, make_density(plus_blocks), options
        ).e_xc
        e_minus = vq.build_xc_periodic(
            basis, system, grid, func, make_density(minus_blocks), options
        ).e_xc
        finite_difference = (e_plus - e_minus) / (2.0 * step)
        projected = sum(
            np.asarray(result.V_xc.blocks[block_index])[row, column]
            for block_index, row, column in probe
        )
        assert projected == pytest.approx(
            finite_difference,
            rel=3.0e-6,
            abs=3.0e-7,
        )


def test_external_periodic_uks_roks_shared_kernel_matches_restricted_singlet():
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis, grid, options, home, make_density = _h2_periodic_fixture()
    total = 0.4 * np.eye(basis.nbasis)

    rks = vq.build_xc_periodic(
        basis,
        system,
        grid,
        vq.Functional(name, 1),
        make_density(total),
        options,
    )
    uks = vq.build_xc_periodic_uks(
        basis,
        system,
        grid,
        vq.Functional(name, 2),
        make_density(0.5 * total),
        make_density(0.5 * total),
        options,
    )

    assert uks.e_xc == pytest.approx(rks.e_xc, abs=2.0e-12)
    np.testing.assert_allclose(
        np.asarray(uks.V_alpha.blocks[home]),
        np.asarray(rks.V_xc.blocks[home]),
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    np.testing.assert_allclose(
        np.asarray(uks.V_beta.blocks[home]),
        np.asarray(rks.V_xc.blocks[home]),
        atol=2.0e-11,
        rtol=2.0e-11,
    )

    restricted_payload = provider.calls[0]
    np.testing.assert_allclose(
        np.asarray(restricted_payload["rho_alpha"]),
        np.asarray(restricted_payload["rho_beta"]),
    )


def test_external_molecular_rks_uks_and_roks_singlet_paths_agree():
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    molecule = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, -0.7]), vq.Atom(1, [0.0, 0.0, 0.7])],
        0,
        1,
    )
    basis = vq.BasisSet(molecule, "sto-3g")

    rks_options = vq.RKSOptions()
    rks_options.functional = name
    rks_options.grid.n_radial = 6
    rks_options.grid.n_theta = 4
    rks_options.grid.n_phi = 6
    rks_options.max_iter = 50
    rks_options.conv_tol_energy = 1.0e-10
    rks_options.conv_tol_grad = 1.0e-8

    uks_options = vq.UKSOptions()
    uks_options.functional = name
    uks_options.grid.n_radial = 6
    uks_options.grid.n_theta = 4
    uks_options.grid.n_phi = 6
    uks_options.max_iter = 50
    uks_options.conv_tol_energy = 1.0e-10
    uks_options.conv_tol_grad = 1.0e-8
    roks_options = vq.ROKSOptions()
    roks_options.grid = rks_options.grid
    roks_options.max_iter = 50
    roks_options.conv_tol_energy = 1.0e-10
    roks_options.conv_tol_grad = 1.0e-8

    rks = vq.run_rks(molecule, basis, rks_options)
    uks = vq.run_uks(molecule, basis, uks_options)
    roks = vq.run_roks(
        molecule,
        basis,
        roks_options,
        functional=name,
    )
    assert rks.converged and uks.converged and roks.converged
    assert uks.energy == pytest.approx(rks.energy, abs=2.0e-9)
    assert uks.e_xc == pytest.approx(rks.e_xc, abs=2.0e-10)
    assert roks.energy == pytest.approx(rks.energy, abs=2.0e-9)
    assert roks.e_xc == pytest.approx(rks.e_xc, abs=2.0e-10)
    with pytest.raises(RuntimeError, match="full-grid external XC"):
        vq.compute_gradient_rks(molecule, basis, rks, rks_options.grid)
    with pytest.raises(RuntimeError, match="full-grid external XC"):
        vq.compute_gradient_uks(molecule, basis, uks, uks_options.grid)


def test_external_callback_wrong_adjoint_shape_fails_loudly():
    def bad_provider(features: dict[str, object]) -> dict[str, object]:
        n = len(np.asarray(features["grid_weights"]))
        return {
            "energy": 0.0,
            "v_rho_alpha": np.zeros(n - 1),
            "v_rho_beta": np.zeros(n),
            "v_grad_alpha": np.zeros((n, 3)),
            "v_grad_beta": np.zeros((n, 3)),
            "v_tau_alpha": np.zeros(n),
            "v_tau_beta": np.zeros(n),
        }

    name = _external_name(bad_provider)
    system, basis, grid, options, _, make_density = _h2_periodic_fixture()
    with pytest.raises(ValueError, match="wrong full-grid shape"):
        vq.build_xc_periodic(
            basis,
            system,
            grid,
            vq.Functional(name, 1),
            make_density(0.4 * np.eye(basis.nbasis)),
            options,
        )


def test_external_periodic_analytic_gradient_fails_closed():
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis, grid, options, _, make_density = _h2_periodic_fixture()
    with pytest.raises(RuntimeError, match="grid/image/partition response"):
        core.xc_lattice_gradient_contribution(
            basis,
            system,
            grid,
            vq.Functional(name, 1),
            make_density(0.4 * np.eye(basis.nbasis)),
            options,
        )


def _h2_molecular_gate_fixture():
    molecule = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, -0.7]), vq.Atom(1, [0.0, 0.0, 0.7])],
    )
    return molecule


def _h2_periodic_gate_fixture():
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [4.0, 4.0, 3.3]), vq.Atom(1, [4.0, 4.0, 4.7])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _large_box_external_scf_fixture(spin: int):
    if spin == 1:
        atoms = [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.4]),
        ]
        multiplicity = 1
    else:
        atoms = [vq.Atom(1, [0.0, 0.0, 0.0])]
        multiplicity = 2
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 20.0,
        atoms,
        0,
        multiplicity,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = core.PeriodicKSOptions()
    options.max_iter = 1
    options.use_diis = False
    options.grid.n_radial = 4
    options.grid.n_theta = 3
    options.grid.n_phi = 4
    options.becke_image_radius_bohr = 6.0
    options.lattice_opts.cutoff_bohr = 6.0
    options.lattice_opts.nuclear_cutoff_bohr = 6.0
    return system, basis, options


@pytest.mark.parametrize(
    ("method", "route_kwargs"),
    (
        ("rks", {"optimize": True}),
        ("rks", {"geom_opt": "rfo"}),
        ("rks", {"hessian": True}),
        ("rks", {"tddft": True}),
        ("rks", {"tddft_gradient": True}),
        ("roks", {"optimize": True}),
    ),
)
def test_external_molecular_response_workflows_fail_before_execution(
    tmp_path,
    method,
    route_kwargs,
):
    name = _external_name(_QuadraticNonlocalProvider())
    with pytest.raises(NotImplementedError, match="full-grid external XC"):
        vq.run_job(
            _h2_molecular_gate_fixture(),
            basis="sto-3g",
            method=method,
            functional=name,
            output=tmp_path / "external-response",
            **route_kwargs,
        )


@pytest.mark.parametrize(
    ("method", "options_type", "threshold"),
    (
        ("rks", vq.RKSOptions, "newton_threshold"),
        ("uks", vq.UKSOptions, "trah_threshold"),
    ),
)
def test_external_molecular_pointwise_second_order_scf_fails_early(
    tmp_path,
    method,
    options_type,
    threshold,
):
    name = _external_name(_QuadraticNonlocalProvider())
    options = options_type()
    options.functional = name
    setattr(options, threshold, 1.0)
    option_kwarg = {f"{method}_options": options}

    with pytest.raises(NotImplementedError, match="pointwise f_xc"):
        vq.run_job(
            _h2_molecular_gate_fixture(),
            basis="sto-3g",
            method=method,
            output=tmp_path / f"external-{method}-second-order",
            **option_kwarg,
        )


def test_generic_external_does_not_inherit_skala_grid_provenance_or_memory(
    tmp_path,
    monkeypatch,
):
    import tomllib

    import vibeqc.runner as runner_module

    name = _external_name(_QuadraticNonlocalProvider())
    molecule = _h2_molecular_gate_fixture()
    basis = vq.BasisSet(molecule, "sto-3g")
    generic_options = vq.RKSOptions()
    generic_options.functional = name
    skala_options = vq.RKSOptions()
    skala_options.functional = "skala-1.1"

    generic_memory = vq.estimate_memory(
        molecule,
        basis,
        method="rks",
        options=generic_options,
    )
    skala_memory = vq.estimate_memory(
        molecule,
        basis,
        method="rks",
        options=skala_options,
    )
    assert (
        skala_memory.by_category["DFT grid + chi"]
        > generic_memory.by_category["DFT grid + chi"]
    )

    applied_levels = []
    real_apply_grid_level = runner_module._apply_grid_level

    def record_grid_level(grid, level):
        applied_levels.append(level)
        return real_apply_grid_level(grid, level)

    monkeypatch.setattr(runner_module, "_apply_grid_level", record_grid_level)
    stem = tmp_path / "generic-external-dry-run"
    result = vq.run_job(
        molecule,
        basis="sto-3g",
        method="rks",
        functional=name,
        output=stem,
        dry_run=True,
        progress=False,
    )
    assert result is None
    assert "orca-defgrid3" in applied_levels
    assert "skala" not in applied_levels
    with stem.with_suffix(".system").open("rb") as handle:
        run_fields = tomllib.load(handle)["run"]
    assert not any(str(key).startswith("skala_") for key in run_fields)


def test_options_only_skala_name_is_recorded_as_effective_functional(tmp_path):
    import tomllib

    options = vq.RKSOptions()
    options.functional = "skala-1.1"
    stem = tmp_path / "skala-options-only-dry-run"

    result = vq.run_job(
        _h2_molecular_gate_fixture(),
        basis="sto-3g",
        method="rks",
        rks_options=options,
        output=stem,
        dry_run=True,
        progress=False,
    )

    assert result is None
    with stem.with_suffix(".system").open("rb") as handle:
        manifest = tomllib.load(handle)
    assert manifest["plan"]["functional"] == "skala-1.1"
    assert manifest["run"]["skala_model_revision"] == vq.SKALA_MODEL_REVISION
    assert (
        manifest["run"]["skala_source_notice_file"]
        == "MICROSOFT-SKALA-SOURCE-MIT.txt"
    )


@pytest.mark.parametrize(
    ("method", "route_kwargs"),
    (
        ("RKS", {"optimize": True}),
        ("RKS", {"optimize": True, "optimize_cell": True}),
        ("RKS", {"hessian": True}),
        ("RKS", {"tddft": True}),
        ("ROKS", {"optimize": True}),
    ),
)
def test_external_periodic_response_workflows_fail_before_execution(
    tmp_path,
    method,
    route_kwargs,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    with pytest.raises(NotImplementedError, match="full-grid external XC"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional=name,
            jk_method="bipole",
            output=tmp_path / "external-periodic-response",
            progress=False,
            **route_kwargs,
        )


def test_external_periodic_roks_single_point_fails_closed(tmp_path):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    stem = tmp_path / "external-periodic-roks"

    with pytest.raises(NotImplementedError, match="periodic ROKS"):
        vq.run_periodic_job(
            system,
            basis,
            method="ROKS",
            functional=name,
            jk_method="bipole",
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("method", ("RKS", "UKS"))
def test_gamma_rijcosx_receives_high_level_semilocal_grid_profile(
    tmp_path,
    monkeypatch,
    method,
):
    import vibeqc.periodic_rijcosx as rijcosx_module

    class StopAtRIJCOSX(RuntimeError):
        pass

    captured = {}

    def capture_driver(_system, _basis, options, **kwargs):
        captured["options"] = options
        captured["grid_options"] = kwargs.get("grid_options")
        raise StopAtRIJCOSX

    monkeypatch.setattr(
        rijcosx_module,
        f"run_periodic_rijcosx_{method.lower()}",
        capture_driver,
    )
    system, basis = _h2_periodic_gate_fixture()

    with pytest.raises(StopAtRIJCOSX):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional="pbe",
            jk_method="rijcosx",
            output=tmp_path / f"{method.lower()}-pbe-rijcosx",
            convergence="off",
            memory_override=True,
            output_qvf=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            citations=False,
            record_hostname=False,
            progress=False,
        )

    assert captured["grid_options"] is captured["options"].grid
    assert (
        captured["grid_options"].atomic_grid_profile
        == vq.AtomicGridProfile.Generic
    )


@pytest.mark.parametrize("method", ("RKS", "UKS"))
def test_external_periodic_gamma_rijcosx_fails_before_dry_run(
    tmp_path,
    method,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    stem = tmp_path / f"external-{method.lower()}-gamma-rijcosx"

    with pytest.raises(
        NotImplementedError,
        match="Gamma/single-k RIJCOSX route",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional=name,
            jk_method="rijcosx",
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("method", ("RKS", "UKS"))
def test_external_periodic_multik_rijcosx_reaches_dry_run(
    tmp_path,
    method,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    stem = tmp_path / f"external-{method.lower()}-multik-rijcosx"

    result = vq.run_periodic_job(
        system,
        basis,
        method=method,
        functional=name,
        jk_method="rijcosx",
        kpoints=(2, 1, 1),
        output=stem,
        dry_run=True,
        progress=False,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("method", "selector"),
    (
        (
            "RKS",
            {
                "jk_method": "aiccm2026dev-b",
                "aiccm_backend": "four_center",
            },
        ),
        (
            "UKS",
            {
                "jk_method": "aiccm2026dev-b",
                "aiccm_backend": "four_center",
            },
        ),
        # Front door (M1): functional=name infers KS; the closed-shell
        # fixture infers RKS (UKS on a singlet is not expressible there).
        (
            "aiccm",
            {"variant": "chi", "aiccm_backend": "four_center"},
        ),
    ),
    ids=("rks-legacy", "uks-legacy", "front-door"),
)
def test_external_periodic_chi_four_center_reaches_dry_run(
    tmp_path,
    method,
    selector,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    stem = tmp_path / f"external-{method.lower()}-chi-ccm"

    result = vq.run_periodic_job(
        system,
        basis,
        method=method,
        functional=name,
        output=stem,
        dry_run=True,
        progress=False,
        **selector,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("method", "selector"),
    (
        (
            "RKS",
            {"jk_method": "aiccm2026dev-b", "aiccm_backend": "ri"},
        ),
        (
            "UKS",
            {"jk_method": "aiccm2026dev-b", "aiccm_backend": "ri"},
        ),
        ("aiccm", {"variant": "chi", "aiccm_backend": "ri"}),
    ),
    ids=("rks-legacy", "uks-legacy", "front-door"),
)
def test_external_periodic_chi_fitted_backend_stays_gated(
    tmp_path,
    method,
    selector,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    stem = tmp_path / f"external-{method.lower()}-chi-ri"

    with pytest.raises(NotImplementedError, match="backend='four_center'"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional=name,
            output=stem,
            dry_run=True,
            progress=False,
            **selector,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("runner_name", "bipole_name"),
    (
        ("run_aiccm2026dev_b_rks", "run_pbc_bipole_rks"),
        ("run_aiccm2026dev_b_uks", "run_pbc_bipole_uks"),
    ),
)
def test_external_chi_direct_call_forces_periodic_domain_and_grid_profile(
    monkeypatch,
    runner_name,
    bipole_name,
):
    import vibeqc.periodic.chi.scf as chi

    class StopBeforeSCF(RuntimeError):
        pass

    seen = {}

    def fake_bipole(system, basis, kmesh, options, **kwargs):
        seen["domain"] = kwargs["xc_density_domain"]
        seen["profile"] = options.grid.atomic_grid_profile
        raise StopBeforeSCF

    monkeypatch.setattr(chi, bipole_name, fake_bipole)
    name = _external_name(
        _QuadraticNonlocalProvider(),
        required_grid_profile="pyscf-level3",
    )
    system, basis = _h2_periodic_gate_fixture()

    with pytest.warns(vq.AICCM2026DevBExperimentalWarning):
        with pytest.raises(StopBeforeSCF):
            getattr(chi, runner_name)(
                system,
                basis,
                name,
                lattice_extension=(1, 1, 1),
                progress=False,
            )

    assert seen["domain"] == vq.PeriodicXCDensityDomain.PERIODIC_LATTICE
    assert str(seen["profile"]).rsplit(".", 1)[-1] == "PySCFLevel3"


@pytest.mark.parametrize(
    ("runner_name", "gdf_name"),
    (
        ("run_aiccm2026dev_b_rks", "run_krks_periodic_gdf"),
        ("run_aiccm2026dev_b_uks", "run_kuks_periodic_gdf"),
    ),
)
def test_chi_fitted_libxc_call_preserves_automatic_density_domain(
    monkeypatch,
    runner_name,
    gdf_name,
):
    import vibeqc.periodic.chi.scf as chi

    class StopBeforeSCF(RuntimeError):
        pass

    seen = {}

    def fake_gdf(*args, **kwargs):
        seen["domain"] = kwargs["xc_density_domain"]
        raise StopBeforeSCF

    monkeypatch.setattr(chi, gdf_name, fake_gdf)
    system, basis = _h2_periodic_gate_fixture()

    with pytest.warns(vq.AICCM2026DevBExperimentalWarning):
        with pytest.raises(StopBeforeSCF):
            getattr(chi, runner_name)(
                system,
                basis,
                "pbe",
                lattice_extension=(2, 1, 1),
                backend="ri",
                progress=False,
            )

    assert seen["domain"] == vq.PeriodicXCDensityDomain.AUTO


@pytest.mark.parametrize(
    "runner_name",
    ("run_krks_periodic_gdf", "run_kuks_periodic_gdf"),
)
def test_external_gdf_direct_call_promotes_gamma_to_periodic_torus(
    monkeypatch,
    runner_name,
):
    import vibeqc.periodic_k_gdf as gdf

    class StopBeforeIntegrals(RuntimeError):
        pass

    seen = {}

    def capture_domain(system, lattice_options, cells, density_domain):
        seen["domain"] = density_domain
        raise StopBeforeIntegrals

    monkeypatch.setattr(gdf, "_xc_density_cells_for_domain", capture_domain)
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    options = vq.PeriodicKSOptions()
    options.functional = name

    with pytest.raises(StopBeforeIntegrals):
        getattr(gdf, runner_name)(
            system,
            basis,
            (1, 1, 1),
            options,
            functional=name,
            progress=False,
        )

    assert seen["domain"] == vq.PeriodicXCDensityDomain.PERIODIC_LATTICE


def test_external_gdf_direct_call_rejects_non_torus_kpoint_list():
    import vibeqc.periodic_k_gdf as gdf

    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    explicit = core.bloch_kmesh_from_lists(
        [np.zeros(3), np.array([0.2, 0.0, 0.0])],
        [0.5, 0.5],
    )

    with pytest.raises(
        NotImplementedError,
        match="complete, uniformly weighted Monkhorst-Pack mesh",
    ):
        gdf.run_krks_periodic_gdf(
            system,
            basis,
            explicit,
            functional=name,
            progress=False,
        )


@pytest.mark.parametrize(
    "runner_name",
    ("run_krks_periodic_gdf", "run_kuks_periodic_gdf"),
)
def test_external_gdf_direct_call_rejects_reduced_mesh_before_expansion(
    monkeypatch,
    runner_name,
):
    import vibeqc.periodic_k_gdf as gdf

    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis = _h2_periodic_gate_fixture()
    vq.attach_symmetry(system)
    reduced = vq.monkhorst_pack(
        system,
        [2, 2, 2],
        use_symmetry=True,
    )
    assert len(reduced.kpoints) < 8
    assert len(reduced.ir_mapping) == 8

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external-XC reduced mesh reached IBZ expansion")

    monkeypatch.setattr(gdf, "_expand_ibz_kmesh_to_full_bz", forbidden)
    monkeypatch.setattr(gdf, "_xc_density_cells_for_domain", forbidden)
    with pytest.raises(
        NotImplementedError,
        match="complete, uniformly weighted Monkhorst-Pack mesh",
    ):
        getattr(gdf, runner_name)(
            system,
            basis,
            reduced,
            functional=name,
            progress=False,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    "runner_name",
    ("run_pbc_gdf_rhf", "run_pbc_gdf_uks"),
)
def test_external_direct_gamma_gdf_fast_paths_fail_closed(runner_name):
    import vibeqc.pbc_gdf as gamma_gdf

    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    options = vq.PeriodicKSOptions()
    options.functional = name

    with pytest.raises(NotImplementedError, match="generic finite-torus"):
        getattr(gamma_gdf, runner_name)(
            system,
            basis,
            options,
            functional=name,
            progress=False,
        )


@pytest.mark.parametrize(
    ("method", "runner_name"),
    (
        ("RKS", "run_krks_periodic_gdf"),
        ("UKS", "run_kuks_periodic_gdf"),
    ),
)
def test_external_high_level_gamma_gdf_uses_finite_torus_driver(
    tmp_path,
    monkeypatch,
    method,
    runner_name,
):
    import vibeqc.periodic_runner as periodic_runner

    class StopBeforeSCF(RuntimeError):
        pass

    captured = {}

    def fake_gdf(_system, _basis, kmesh, _options, **kwargs):
        captured["kmesh"] = tuple(kmesh)
        captured["domain"] = kwargs["xc_density_domain"]
        raise StopBeforeSCF

    monkeypatch.setattr(periodic_runner, runner_name, fake_gdf)
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()

    with pytest.raises(StopBeforeSCF):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional=name,
            jk_method="gdf",
            output=tmp_path / f"external-{method.lower()}-gdf-gamma",
            output_qvf=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            record_hostname=False,
            progress=False,
        )

    assert captured == {
        "kmesh": (1, 1, 1),
        "domain": vq.PeriodicXCDensityDomain.PERIODIC_LATTICE,
    }


@pytest.mark.parametrize("jk_method", ("gdf", "rijcosx"))
def test_external_high_level_fitted_routes_reject_non_torus_mesh_before_dry_run(
    tmp_path,
    jk_method,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    explicit = core.bloch_kmesh_from_lists(
        [np.zeros(3), np.array([0.2, 0.0, 0.0])],
        [0.25, 0.75],
    )
    stem = tmp_path / f"external-{jk_method}-explicit"

    with pytest.raises(
        NotImplementedError,
        match="complete, uniformly weighted Monkhorst-Pack mesh",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional=name,
            jk_method=jk_method,
            kpoints=explicit,
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("jk_method", ("gdf", "bipole", "gpw", "gapw"))
def test_external_high_level_positive_smearing_fails_before_dry_run(
    tmp_path,
    jk_method,
):
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis = _h2_periodic_gate_fixture()
    stem = tmp_path / f"external-{jk_method}-finite-temperature"

    with pytest.raises(NotImplementedError, match="zero-temperature only"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional=name,
            jk_method=jk_method,
            smearing_temperature=0.01,
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert provider.calls == []
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("source", ("options", "kpoints"))
def test_external_high_level_effective_smearing_fails_before_dry_run(
    tmp_path,
    source,
):
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis = _h2_periodic_gate_fixture()
    stem = tmp_path / f"external-gdf-{source}-finite-temperature"
    kwargs = {}
    if source == "options":
        kwargs["smearing"] = vq.SmearingOptions(temperature=0.01)
    else:
        kpoints = vq.KPoints.gamma(system)
        kpoints.smearing = vq.SmearingOptions(temperature=0.01)
        kwargs["kpoints"] = kpoints

    with pytest.raises(NotImplementedError, match="zero-temperature only"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional=name,
            jk_method="gdf",
            output=stem,
            dry_run=True,
            progress=False,
            **kwargs,
        )

    assert provider.calls == []
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("jk_method", ("gpw", "gapw"))
def test_external_periodic_gpw_gapw_rks_routes_accept_gamma_dry_run(
    tmp_path,
    jk_method,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    stem = tmp_path / f"external-{jk_method}"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional=name,
        jk_method=jk_method,
        output=stem,
        dry_run=True,
        progress=False,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize("jk_method", ("gpw", "gapw"))
def test_external_periodic_gpw_gapw_accept_gamma_kpoints_object_dry_run(
    tmp_path,
    jk_method,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    kpoints = vq.KPoints.gamma(system)
    stem = tmp_path / f"external-{jk_method}-gamma-object"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional=name,
        jk_method=jk_method,
        kpoints=kpoints,
        output=stem,
        dry_run=True,
        progress=False,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize("native_bloch_mesh", (False, True))
def test_external_gapw_gamma_object_plus_u_plans_gamma_route(
    tmp_path,
    native_bloch_mesh,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    kpoints = vq.KPoints.gamma(system)
    if native_bloch_mesh:
        kpoints = kpoints.to_bloch_kmesh()
    stem = tmp_path / (
        "external-gapw-gamma-bloch-plus-u"
        if native_bloch_mesh
        else "external-gapw-gamma-kpoints-plus-u"
    )

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional=name,
        jk_method="gapw",
        kpoints=kpoints,
        dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
        output=stem,
        output_qvf=False,
        dry_run=True,
        progress=False,
    )

    assert result is None
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["run"]["dft_plus_u_route"] == "gapw_rks_gamma"


@pytest.mark.parametrize(
    "jk_method",
    (
        "gdf",
        "rijcosx",
        "gpw",
        "gapw",
        "real-gamma",
        "aiccm2026dev-a",
        "aiccm2026dev-b",
    ),
)
def test_external_selected_route_ecp_rejected_before_dry_run(
    tmp_path,
    monkeypatch,
    jk_method,
):
    import vibeqc.periodic_runner as periodic_runner

    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    stem = tmp_path / f"external-{jk_method}-ecp"
    monkeypatch.setattr(
        periodic_runner,
        "_resolve_ecp_data",
        lambda _system, _basis: ([object()], [0], [1.0, 1.0], 2),
    )

    with pytest.raises(NotImplementedError, match="ECP-bearing bases"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional=name,
            jk_method=jk_method,
            kpoints=((2, 1, 1) if jk_method == "rijcosx" else None),
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    "jk_method",
    (
        "gdf",
        "rijcosx",
        "gpw",
        "gapw",
        "real-gamma",
        "aiccm2026dev-a",
        "aiccm2026dev-b",
    ),
)
def test_external_ecp_paired_basis_name_rejected_before_dry_run(
    tmp_path,
    jk_method,
):
    name = _external_name(_QuadraticNonlocalProvider())
    # LANL2DZ is all-electron for hydrogen; use an actual ECP-bearing atom.
    system = vq.PeriodicSystem(
        3, np.eye(3) * 8.0, [vq.Atom(30, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    stem = tmp_path / f"external-{jk_method}-paired-basis"

    with pytest.raises(NotImplementedError, match="ECP-bearing bases"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional=name,
            jk_method=jk_method,
            kpoints=((2, 1, 1) if jk_method == "rijcosx" else None),
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("spin", (1, 2))
def test_external_gdf_low_level_rejects_ecp_paired_basis(spin):
    import vibeqc.periodic_k_gdf as gdf_module

    name = _external_name(_QuadraticNonlocalProvider())
    system = vq.PeriodicSystem(
        3, np.eye(3) * 8.0, [vq.Atom(30, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")

    with pytest.raises(NotImplementedError, match="ECP-bearing"):
        if spin == 1:
            gdf_module.run_krks_periodic_gdf(
                system,
                basis,
                functional=name,
            )
        else:
            gdf_module.run_kuks_periodic_gdf(
                system,
                basis,
                functional=name,
            )


def test_external_periodic_two_dimensional_route_rejected_before_dry_run(
    tmp_path,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system = vq.PeriodicSystem(
        2,
        np.eye(3) * 12.0,
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [1.4, 0.0, 0.0]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "external-two-dimensional"

    with pytest.raises(NotImplementedError, match="2D external XC remains gated"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional=name,
            jk_method="gdf",
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("spin", (1, 2))
def test_external_aiccm_rejects_molecular_becke_partition_before_integrals(
    spin,
):
    from vibeqc.periodic.chi.scf import (
        run_aiccm2026dev_b_rks,
        run_aiccm2026dev_b_uks,
    )

    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    opts = core.PeriodicKSOptions()
    opts.use_periodic_becke = False
    driver = (
        run_aiccm2026dev_b_rks if spin == 1 else run_aiccm2026dev_b_uks
    )

    with pytest.raises(ValueError, match="use_periodic_becke=True"):
        driver(
            system,
            basis,
            name,
            mesh=(1, 1, 1),
            options=opts,
            backend="four_center",
        )


@pytest.mark.parametrize(
    ("module_name", "driver_name", "spin"),
    (
        ("vibeqc.pbc_bipole_rks", "run_pbc_bipole_rks", 1),
        ("vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks", 2),
        ("vibeqc.periodic_k_gdf", "run_krhf_periodic_gdf", 1),
        ("vibeqc.periodic_k_gdf", "run_kuhf_periodic_gdf", 2),
    ),
)
def test_direct_periodic_external_routes_select_required_grid_profile(
    monkeypatch,
    module_name,
    driver_name,
    spin,
):
    import importlib

    class GridCaptured(RuntimeError):
        pass

    name = _external_name(
        _QuadraticNonlocalProvider(),
        required_grid_profile="pyscf-level3",
    )
    system, basis = _h2_periodic_gate_fixture()
    module = importlib.import_module(module_name)
    captured = []

    def capture_grid(_system, *, grid_options, **_kwargs):
        captured.append(grid_options.atomic_grid_profile)
        raise GridCaptured

    monkeypatch.setattr(module, "build_periodic_becke_grid", capture_grid)
    driver = getattr(module, driver_name)
    kmesh = core.monkhorst_pack(system, [2, 1, 1])
    kwargs = {
        "functional": name,
        "xc_density_domain": core.PeriodicXCDensityDomain.PERIODIC_LATTICE,
    }
    with pytest.raises(GridCaptured):
        driver(system, basis, kmesh, **kwargs)

    required = core.GridOptions()
    required.atomic_grid_profile = "pyscf-level3"
    assert captured == [required.atomic_grid_profile]


@pytest.mark.parametrize(
    ("module_name", "driver_name"),
    (
        ("vibeqc.pbc_bipole_rks", "run_pbc_bipole_rks"),
        ("vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks"),
        ("vibeqc.periodic_k_gdf", "run_krhf_periodic_gdf"),
        ("vibeqc.periodic_k_gdf", "run_kuhf_periodic_gdf"),
    ),
)
def test_direct_periodic_external_routes_reject_explicit_grid_mismatch(
    module_name,
    driver_name,
):
    import importlib

    name = _external_name(
        _QuadraticNonlocalProvider(),
        required_grid_profile="pyscf-level3",
    )
    system, basis = _h2_periodic_gate_fixture()
    opts = core.PeriodicKSOptions()
    opts.grid.atomic_grid_profile = "generic"
    driver = getattr(importlib.import_module(module_name), driver_name)
    kmesh = core.monkhorst_pack(system, [2, 1, 1])

    with pytest.raises(ValueError, match="requires grid profile 'pyscf-level3'"):
        driver(
            system,
            basis,
            kmesh,
            options=opts,
            functional=name,
            xc_density_domain=(
                core.PeriodicXCDensityDomain.PERIODIC_LATTICE
            ),
        )


@pytest.mark.parametrize(
    ("module_name", "driver_name"),
    (
        ("vibeqc.pbc_bipole_rks", "run_pbc_bipole_rks"),
        ("vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks"),
        ("vibeqc.periodic_k_gdf", "run_krhf_periodic_gdf"),
        ("vibeqc.periodic_k_gdf", "run_kuhf_periodic_gdf"),
    ),
)
def test_direct_periodic_external_routes_reject_positive_smearing_before_grid(
    monkeypatch,
    module_name,
    driver_name,
):
    import importlib

    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis = _h2_periodic_gate_fixture()
    options = core.PeriodicKSOptions()
    options.smearing_temperature = 0.01
    module = importlib.import_module(module_name)

    def forbidden_grid(*_args, **_kwargs):
        raise AssertionError("external-XC smearing guard ran after grid setup")

    monkeypatch.setattr(module, "build_periodic_becke_grid", forbidden_grid)
    with pytest.raises(NotImplementedError, match="zero-temperature only"):
        getattr(module, driver_name)(
            system,
            basis,
            core.monkhorst_pack(system, [2, 1, 1]),
            options,
            functional=name,
            xc_density_domain=(
                core.PeriodicXCDensityDomain.PERIODIC_LATTICE
            ),
            progress=False,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    ("module_name", "driver_name"),
    (
        ("vibeqc.pbc_bipole_rks", "run_pbc_bipole_rks"),
        ("vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks"),
    ),
)
def test_direct_bipole_external_rejects_legacy_gauge_before_grid(
    monkeypatch,
    module_name,
    driver_name,
):
    import importlib

    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis = _h2_periodic_gate_fixture()
    module = importlib.import_module(module_name)

    def forbidden_grid(*_args, **_kwargs):
        raise AssertionError("external-XC gauge guard ran after grid setup")

    monkeypatch.setattr(module, "build_periodic_becke_grid", forbidden_grid)
    with pytest.raises(
        NotImplementedError,
        match="requires the corrected Ewald exchange gauge",
    ):
        getattr(module, driver_name)(
            system,
            basis,
            core.monkhorst_pack(system, [1, 1, 1]),
            functional=name,
            xc_density_domain=(
                core.PeriodicXCDensityDomain.PERIODIC_LATTICE
            ),
            use_exchange_ewald_split=False,
            progress=False,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    ("module_name", "driver_name"),
    (
        ("vibeqc.pbc_bipole_rks", "run_pbc_bipole_rks"),
        ("vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks"),
    ),
)
def test_direct_bipole_external_rejects_ad_hoc_multik_before_grid(
    monkeypatch,
    module_name,
    driver_name,
):
    import importlib

    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis = _h2_periodic_gate_fixture()
    module = importlib.import_module(module_name)
    explicit = core.bloch_kmesh_from_lists(
        [np.zeros(3), np.array([0.2, 0.0, 0.0])],
        [0.5, 0.5],
    )

    def forbidden_grid(*_args, **_kwargs):
        raise AssertionError("external-XC mesh guard ran after grid setup")

    monkeypatch.setattr(module, "build_periodic_becke_grid", forbidden_grid)
    with pytest.raises(
        NotImplementedError,
        match="complete Monkhorst-Pack mesh",
    ):
        getattr(module, driver_name)(
            system,
            basis,
            explicit,
            functional=name,
            xc_density_domain=(
                core.PeriodicXCDensityDomain.PERIODIC_LATTICE
            ),
            progress=False,
        )

    assert provider.calls == []


def _assert_periodic_ks_operator_channels(result, spin: int) -> None:
    if spin == 1:
        channels = ((result.fock, result.mo_coeffs, result.mo_energies),)
    else:
        channels = (
            (
                result.fock_alpha,
                result.mo_coeffs_alpha,
                result.mo_energies_alpha,
            ),
            (
                result.fock_beta,
                result.mo_coeffs_beta,
                result.mo_energies_beta,
            ),
        )
    for focks, coeffs, energies in channels:
        for fock_k, overlap_k, coeffs_k, energies_k in zip(
            focks,
            result.overlap,
            coeffs,
            energies,
        ):
            residual = np.asarray(fock_k) @ np.asarray(coeffs_k) - (
                np.asarray(overlap_k) @ np.asarray(coeffs_k)
            ) * np.asarray(energies_k)[None, :]
            assert float(np.max(np.abs(residual), initial=0.0)) < 5.0e-9


@pytest.mark.parametrize("spin", (1, 2))
def test_direct_bipole_external_scf_is_live_and_returns_physical_focks(spin):
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis, options = _large_box_external_scf_fixture(spin)
    kmesh = core.monkhorst_pack(system, [1, 1, 1])
    driver = run_pbc_bipole_rks if spin == 1 else run_pbc_bipole_uks

    result = driver(
        system,
        basis,
        kmesh,
        options,
        functional=name,
        progress=False,
        xc_density_domain=core.PeriodicXCDensityDomain.PERIODIC_LATTICE,
    )

    assert result.n_iter == 1
    assert np.isfinite(result.energy)
    assert np.isfinite(result.e_xc)
    assert provider.calls
    assert provider.calls[0]["periodic"] is True
    assert provider.calls[0]["periodic_dimension"] == 3
    _assert_periodic_ks_operator_channels(result, spin)


@pytest.mark.parametrize("spin", (1, 2))
def test_chi_four_center_external_scf_is_live_and_returns_physical_focks(spin):
    from vibeqc.periodic.chi.scf import (
        run_aiccm2026dev_b_rks,
        run_aiccm2026dev_b_uks,
    )

    provider = _QuadraticNonlocalProvider(scale=0.01)
    name = _external_name(provider)
    system, basis, options = _large_box_external_scf_fixture(spin)
    driver = (
        run_aiccm2026dev_b_rks
        if spin == 1
        else run_aiccm2026dev_b_uks
    )

    result = driver(
        system,
        basis,
        name,
        lattice_extension=(1, 1, 1),
        options=options,
        backend="four_center",
        progress=False,
    )

    assert result.n_iter == 1
    assert result.backend == "aiccm2026dev-b-four_center"
    assert np.isfinite(result.energy)
    assert np.isfinite(result.e_xc)
    assert provider.calls
    assert provider.calls[0]["periodic"] is True
    assert provider.calls[0]["periodic_dimension"] == 3
    _assert_periodic_ks_operator_channels(result, spin)


@pytest.mark.parametrize(
    ("spin", "reference"),
    ((1, "rks"), (2, "uks")),
    ids=("rks", "uks"),
)
def test_four_center_external_front_door_reaches_live_scf(
    tmp_path,
    spin,
    reference,
):
    """The AICCM front door must preserve the literal-WSSC XC provider.

    The low-level RKS/UKS variational and operator contracts are pinned in
    ``test_ccm_dft.py``.  This route-level check closes the remaining seam:
    method/reference inference, external-functional identity, provider
    execution, and the per-unit-cell four-center adapter all survive the
    unified runner without loading Torch or substituting libxc.
    """
    provider = _QuadraticNonlocalProvider(scale=0.01)
    name = _external_name(provider)
    system, basis, _ = _large_box_external_scf_fixture(spin)
    stem = tmp_path / f"external-four-center-{reference}"

    result = vq.run_periodic_job(
        system,
        basis,
        method="aiccm",
        variant="four-center",
        functional=name,
        aiccm_lattice_extension=(1, 1, 1),
        max_iter=20,
        output=stem,
        output_qvf=False,
        write_xyz_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        progress=False,
    )

    assert result.converged
    assert 1 < result.n_iter <= 20
    assert np.isfinite(result.energy)
    assert np.isfinite(result.e_xc)
    assert result.functional == name
    assert result.backend == "ccm-four-center"
    assert result.four_center.weighting == "aiccm2026dev-a"
    assert result.ccm_result.backend.endswith(
        f"-aiccm2026dev-a-{reference}"
    )
    assert provider.calls
    assert provider.calls[0]["periodic"] is True
    assert provider.calls[0]["periodic_dimension"] == 3
    if spin == 1:
        assert result.fock_beta is None
        assert result.density_alpha is None
        assert result.density_beta is None
    else:
        assert result.fock_beta is not None
        assert result.density_alpha is not None
        assert result.density_beta is not None


def test_four_center_front_door_forwards_external_grid_controls(
    tmp_path,
    monkeypatch,
):
    """The unified route must not replace provider grid/domain controls."""
    from vibeqc.periodic_external_xc import PeriodicExternalXC

    name = _external_name(
        _QuadraticNonlocalProvider(),
        required_grid_profile="pyscf-level3",
    )
    system, basis, _ = _large_box_external_scf_fixture(1)
    captured = {}

    class DispatchCaptured(Exception):
        pass

    def capture_prepare(cls, *args, **kwargs):
        captured.update(kwargs)
        raise DispatchCaptured

    monkeypatch.setattr(
        PeriodicExternalXC,
        "prepare",
        classmethod(capture_prepare),
    )
    with pytest.raises(DispatchCaptured):
        vq.run_periodic_job(
            system,
            basis,
            method="aiccm",
            variant="four-center",
            functional=name,
            aiccm_lattice_extension=(1, 1, 1),
            output=tmp_path / "external-four-center-grid-controls",
            output_qvf=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            progress=False,
        )

    assert captured["grid_options"].atomic_grid_profile == (
        vq.AtomicGridProfile.PySCFLevel3
    )
    assert captured["image_radius_bohr"] == pytest.approx(10.0)
    assert captured["lattice_options"] is not None


def test_four_center_external_hybrid_rejected_before_dry_run(tmp_path):
    """The public route applies the low-level pure-provider gate preflight."""
    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider, hf_exchange_fraction=0.25)
    system, basis, _ = _large_box_external_scf_fixture(1)
    stem = tmp_path / "external-four-center-hybrid"

    with pytest.raises(NotImplementedError, match="pure full-grid"):
        vq.run_periodic_job(
            system,
            basis,
            method="aiccm",
            variant="four-center",
            functional=name,
            aiccm_lattice_extension=(1, 1, 1),
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert provider.calls == []
    assert not stem.with_suffix(".system").exists()


def test_direct_gdf_external_pure_xc_reports_final_hartree_component(monkeypatch):
    from vibeqc import periodic_k_gdf as driver

    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    evaluated_densities = []
    build_xc = driver._build_xc_k_from_density

    def capture_xc(**kwargs):
        evaluated_densities.append(np.asarray(kwargs['density_k']).copy())
        return build_xc(**kwargs)

    monkeypatch.setattr(driver, '_build_xc_k_from_density', capture_xc)
    system, basis = _h2_periodic_gate_fixture()
    opts = core.PeriodicKSOptions()
    opts.max_iter = 1
    opts.use_diis = False
    opts.grid.n_radial = 3
    opts.grid.n_theta = 2
    opts.grid.n_phi = 3
    opts.becke_image_radius_bohr = 2.0
    opts.lattice_opts.cutoff_bohr = 2.0
    opts.lattice_opts.nuclear_cutoff_bohr = 2.0

    result = driver.run_krhf_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        opts,
        functional=name,
        use_compcell=False,
        check_energy_sanity=False,
        progress=False,
        xc_density_domain=core.PeriodicXCDensityDomain.PERIODIC_LATTICE,
    )

    e_hcore = sum(
        float(weight)
        * float(
            np.real(
                np.trace(np.asarray(density) @ np.asarray(hcore))
            )
        )
        for weight, density, hcore in zip(
            result.kpoint_weights,
            result.density,
            result.hcore,
        )
    )
    assert abs(result.e_coulomb) > 1.0e-8
    assert result.e_hf_exchange == pytest.approx(0.0, abs=1.0e-14)
    # E_electronic includes the functional energy. Reconstruct it from
    # the provider's quadrature inputs, not from the returned component
    # itself. The old assertion omitted this nonzero term (#213).
    features = provider.calls[-1]
    integrated_density = float(np.asarray(features['grid_weights']) @ (
        np.asarray(features['rho_alpha']) + np.asarray(features['rho_beta'])))
    expected_xc = .5 * provider.scale * integrated_density**2
    assert expected_xc > 1.0e-8
    assert result.e_xc == pytest.approx(expected_xc, abs=1.0e-11)
    np.testing.assert_allclose(result.density, evaluated_densities[-1], atol=1e-12, rtol=0.)
    assert result.e_electronic == pytest.approx(
        e_hcore + result.e_coulomb + result.e_hf_exchange + expected_xc,
        abs=1.0e-11,
    )
    assert result.energy == pytest.approx(result.e_electronic + result.e_nuclear, abs=1.0e-11)


def test_direct_gdf_external_uks_multik_returns_physical_spin_focks():
    from vibeqc.periodic_k_gdf import run_kuks_periodic_gdf

    provider = _QuadraticNonlocalProvider()
    name = _external_name(provider)
    system, basis = _h2_periodic_gate_fixture()
    opts = core.PeriodicKSOptions()
    opts.max_iter = 1
    opts.use_diis = False
    opts.grid.n_radial = 3
    opts.grid.n_theta = 2
    opts.grid.n_phi = 3
    opts.becke_image_radius_bohr = 2.0
    opts.lattice_opts.cutoff_bohr = 2.0
    opts.lattice_opts.nuclear_cutoff_bohr = 2.0

    result = run_kuks_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        opts,
        functional=name,
        check_energy_sanity=False,
        progress=False,
        xc_density_domain=core.PeriodicXCDensityDomain.PERIODIC_LATTICE,
    )

    assert result.n_iter == 1
    assert np.isfinite(result.energy)
    assert np.isfinite(result.e_xc)
    assert provider.calls
    for focks, coeffs, energies in (
        (
            result.fock_alpha,
            result.mo_coeffs_alpha,
            result.mo_energies_alpha,
        ),
        (
            result.fock_beta,
            result.mo_coeffs_beta,
            result.mo_energies_beta,
        ),
    ):
        for fock_k, overlap_k, coeffs_k, energies_k in zip(
            focks,
            result.overlap,
            coeffs,
            energies,
        ):
            residual = np.asarray(fock_k) @ np.asarray(coeffs_k) - (
                np.asarray(overlap_k) @ np.asarray(coeffs_k)
            ) * np.asarray(energies_k)[None, :]
            assert float(np.max(np.abs(residual), initial=0.0)) < 5.0e-9


def test_external_grid_profile_is_negotiated_before_periodic_dry_run(
    tmp_path,
    monkeypatch,
):
    import vibeqc.periodic_runner as periodic_runner

    name = _external_name(
        _QuadraticNonlocalProvider(),
        required_grid_profile="pyscf-level3",
    )
    original = periodic_runner._negotiate_external_xc_grid_profile
    negotiated = []

    def capture(functional):
        profile = original(functional)
        negotiated.append(profile)
        return profile

    monkeypatch.setattr(
        periodic_runner,
        "_negotiate_external_xc_grid_profile",
        capture,
    )
    system, basis = _h2_periodic_gate_fixture()

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional=name,
        jk_method="gapw",
        output=tmp_path / "external-profile-dry-run",
        dry_run=True,
        progress=False,
    )

    assert result is None
    assert negotiated == [vq.AtomicGridProfile.PySCFLevel3]


def test_external_grid_profile_reaches_high_level_gapw_options(
    tmp_path,
    monkeypatch,
):
    import vibeqc.periodic_gapw_augment as gapw

    class StopBeforeSCF(RuntimeError):
        pass

    captured = {}

    def fake_gapw(*args, **kwargs):
        captured["profile"] = kwargs[
            "external_xc_grid_options"
        ].atomic_grid_profile
        raise StopBeforeSCF

    monkeypatch.setattr(gapw, "run_periodic_rhf_gapw", fake_gapw)
    name = _external_name(
        _QuadraticNonlocalProvider(),
        required_grid_profile="pyscf-level3",
    )
    system, basis = _h2_periodic_gate_fixture()

    with pytest.raises(StopBeforeSCF):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional=name,
            jk_method="gapw",
            output=tmp_path / "external-profile-live",
            output_qvf=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            record_hostname=False,
            progress=False,
        )

    assert captured["profile"] == vq.AtomicGridProfile.PySCFLevel3


def test_external_periodic_gpw_accepts_full_mp_object_before_dry_run(
    tmp_path,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    kpoints = vq.KPoints.monkhorst_pack(
        system,
        (2, 1, 1),
        symmetry=False,
    )
    stem = tmp_path / "external-gpw-full-mp"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional=name,
        jk_method="gpw",
        kpoints=kpoints,
        output=stem,
        dry_run=True,
        progress=False,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()


def test_external_periodic_gpw_rejects_explicit_weighted_list_before_dry_run(
    tmp_path,
):
    name = _external_name(_QuadraticNonlocalProvider())
    system, basis = _h2_periodic_gate_fixture()
    explicit = core.bloch_kmesh_from_lists(
        [np.zeros(3), np.array([0.2, 0.0, 0.0])],
        [0.25, 0.75],
    )
    stem = tmp_path / "external-gpw-explicit"

    with pytest.raises(
        NotImplementedError,
        match="complete, uniformly weighted Monkhorst-Pack mesh",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional=name,
            jk_method="gpw",
            kpoints=explicit,
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert not stem.with_suffix(".system").exists()
