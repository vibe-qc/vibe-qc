"""Focused tests for full-grid external XC on the GPW RKS routes."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc import periodic_external_xc as external_xc
from vibeqc.periodic_external_xc import (
    PeriodicExternalXC,
    _reject_reduced_kmesh,
)
from vibeqc.periodic_gapw_grid import PlaneWaveGrid

from tests.refusal_receipt import assert_preflight_refusal


_NAMES = itertools.count()


class _QuadraticDensityProvider:
    """Analytic non-local test functional ``E = 1/2 (int rho)^2``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, features: dict[str, object]) -> dict[str, object]:
        self.calls.append(features)
        weights = np.asarray(features["grid_weights"], dtype=float)
        rho = np.asarray(features["rho_alpha"], dtype=float) + np.asarray(
            features["rho_beta"], dtype=float
        )
        charge = float(weights @ rho)
        q_rho = charge * weights
        zeros = np.zeros(weights.size)
        zeros_grad = np.zeros((weights.size, 3))
        return {
            "energy": 0.5 * charge * charge,
            "v_rho_alpha": q_rho,
            "v_rho_beta": q_rho,
            "v_grad_alpha": zeros_grad,
            "v_grad_beta": zeros_grad,
            "v_tau_alpha": zeros,
            "v_tau_beta": zeros,
        }


def _functional(provider, *, required_grid_profile: str = "", hf_exchange_fraction=0.0):
    name = f"test-gpw-full-grid-xc-{next(_NAMES)}"
    vq.define_external_functional(
        name,
        provider,
        required_grid_profile=required_grid_profile,
        hf_exchange_fraction=hf_exchange_fraction,
    )
    return name, vq.Functional(name, 1)


def _system_and_basis(length: float = 8.0):
    atoms = [
        vq.Atom(1, [length / 2, length / 2, length / 2 - 0.7]),
        vq.Atom(1, [length / 2, length / 2, length / 2 + 0.7]),
    ]
    system = vq.PeriodicSystem(3, np.eye(3) * length, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _ecp_bearing_system_and_basis(length: float = 8.0):
    """A cell whose atoms carry LANL2DZ ECPs (Na: 10 core, Cl: 10 core).

    The ECP decision is per element (registry + sidecar), never per basis
    name, so a hydrogen cell in LANL2DZ is a correct all-electron D95V
    calculation and must not be refused; the guards under test fire for
    atoms that replace core electrons."""
    atoms = [
        vq.Atom(11, [length / 2, length / 2, length / 2 - 1.2]),
        vq.Atom(17, [length / 2, length / 2, length / 2 + 1.2]),
    ]
    system = vq.PeriodicSystem(3, np.eye(3) * length, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    return system, basis


def _coarse_atom_grid_options():
    options = vq.GridOptions()
    options.n_radial = 4
    options.n_theta = 3
    options.n_phi = 4
    return options


def _short_lattice_options(cutoff: float = 2.0):
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = cutoff
    options.nuclear_cutoff_bohr = cutoff
    return options


def _assert_generalized_eigenpairs(fock, overlap, coeffs, energies):
    fock = np.asarray(fock)
    overlap = np.asarray(overlap)
    coeffs = np.asarray(coeffs)
    energies = np.asarray(energies)
    residual = fock @ coeffs - (overlap @ coeffs) * energies[None, :]
    assert float(np.max(np.abs(residual), initial=0.0)) < 5.0e-10


@pytest.mark.parametrize("bitmap_budget", [0, 1 << 20])
@pytest.mark.parametrize("dimension", [0, 1, 2, 3])
def test_lattice_difference_support_matches_explicit_pairs(
    monkeypatch, bitmap_budget, dimension,
):
    monkeypatch.setattr(external_xc, "_DIFFERENCE_BITMAP_MAX_BYTES", bitmap_budget)
    rng = np.random.default_rng(120 + dimension)
    ket = rng.integers(-3, 4, size=(29, 3))
    bra = rng.integers(-2, 3, size=(17, 3))
    ket[:, dimension:] = 0
    bra[:, dimension:] = 0
    # Unequal, gapped supports, duplicates and independent large translations.
    ket = [tuple(map(int, row)) for row in ket + [101, -31, 7]]
    bra = [tuple(map(int, row)) for row in bra + [-23, 13, 91]]
    expected = {
        tuple(s[i] - a[i] for i in range(3)) for s in ket for a in bra
    }
    actual = external_xc._lattice_difference_set(ket, bra)
    assert len(actual) == len(expected)
    assert isinstance(actual, external_xc._LatticeDifferenceBitmap) == bool(
        bitmap_budget
    )
    # Probe the whole box, holes and invalid coordinates that would alias if
    # flattened without per-axis range checks.
    axes = [
        range(min(g[i] for g in expected) - 2, max(g[i] for g in expected) + 3)
        for i in range(3)
    ]
    for index in itertools.product(*axes):
        assert (index in actual) == (index in expected)


@pytest.mark.parametrize(
    "ket,bra", [([], []), ([], [(0, 0, 0)]), ([(0, 0, 0)], [])],
)
def test_lattice_difference_support_empty(ket, bra):
    assert external_xc._lattice_difference_set(ket, bra) == frozenset()


@pytest.mark.parametrize("spacing", [20, 10**30])
def test_lattice_difference_support_sparse_fallback(spacing):
    ket = [(0, 0, 0), (spacing, spacing, spacing)]
    bra = [(0, 0, 0), (1, -2, 3)]
    actual = external_xc._lattice_difference_set(ket, bra)
    assert isinstance(actual, frozenset)
    assert actual == {
        (0, 0, 0), (-1, 2, -3),
        (spacing, spacing, spacing), (spacing - 1, spacing + 2, spacing - 3),
    }


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_external_xc_difference_domain_matches_explicit_order(dimension):
    from vibeqc.pair_resolved_truncation import (
        atom_pair_span, pair_resolved_domain,
    )

    lattice = np.array([[4.0, 0.0, 0.0], [1.2, 3.9, 0.0], [-0.6, 0.8, 4.1]])
    system = vq.PeriodicSystem(dimension, lattice, [
        vq.Atom(1, [0.1, 0.2, 0.3]), vq.Atom(1, [4.3, 0.4, 0.2]),
    ])
    cutoff, radius = 4.2, 2.3
    ket = pair_resolved_domain(system, cutoff).cells
    bra = pair_resolved_domain(system, radius).cells
    required = {
        tuple((np.asarray(s.index) - np.asarray(a.index)).tolist())
        for s in ket for a in bra
    }
    candidates = core.direct_lattice_cells(
        system, cutoff + radius + 2.0 * atom_pair_span(system) + 1.0e-9,
    )
    expected = [
        tuple(cell.index) for cell in candidates if tuple(cell.index) in required
    ]
    actual = external_xc.periodic_xc_difference_closed_cells(system, cutoff, radius)
    assert [tuple(cell.index) for cell in actual] == expected
    assert set(expected) == required


@pytest.mark.parametrize("bitmap_budget", [0, 1 << 20])
def test_external_xc_difference_domain_counts_missing_unique_cells(
    monkeypatch, bitmap_budget,
):
    from vibeqc import pair_resolved_truncation as domains

    cells = [SimpleNamespace(index=(i, 0, 0)) for i in (0, 1)]
    monkeypatch.setattr(external_xc, "_DIFFERENCE_BITMAP_MAX_BYTES", bitmap_budget)
    monkeypatch.setattr(
        domains, "pair_resolved_domain", lambda *args: SimpleNamespace(cells=cells),
    )
    monkeypatch.setattr(domains, "atom_pair_span", lambda *args: 0.0)
    # Duplicate available candidates must not hide the two missing cells.
    monkeypatch.setattr(
        core, "direct_lattice_cells", lambda *args: [cells[0], cells[0]],
    )
    with pytest.raises(RuntimeError, match=r"2 lattice cells missing"):
        external_xc.periodic_xc_difference_closed_cells(object(), 1.0, 1.0)
    ordered = [cells[1], cells[0], cells[0], SimpleNamespace(index=(-1, 0, 0))]
    monkeypatch.setattr(core, "direct_lattice_cells", lambda *args: ordered)
    actual = external_xc.periodic_xc_difference_closed_cells(object(), 1.0, 1.0)
    assert actual == tuple(ordered)


@pytest.mark.parametrize("partition_radius", [2.0, 4.1, 6.0])
def test_external_xc_adapter_domain_is_difference_closed_and_variational(partition_radius):
    from vibeqc.pair_resolved_truncation import pair_resolved_domain

    provider = _QuadraticDensityProvider()
    _, functional = _functional(provider)
    system, basis = _system_and_basis(length=4.0)
    lattice_options = _short_lattice_options(cutoff=4.1)

    context = PeriodicExternalXC.prepare(
        basis,
        system,
        functional,
        grid_options=_coarse_atom_grid_options(),
        image_radius_bohr=partition_radius,
        lattice_options=lattice_options,
    )

    active = pair_resolved_domain(
        system, lattice_options.cutoff_bohr
    ).cells
    output_indices = {
        tuple(np.asarray(cell.index, dtype=int).tolist())
        for cell in context.cells
    }
    for bra in active:
        for ket in active:
            difference = tuple(
                (
                    np.asarray(ket.index, dtype=int)
                    - np.asarray(bra.index, dtype=int)
                ).tolist()
            )
            assert difference in output_indices

    density = np.array([[0.7, 0.1], [0.1, 0.5]])
    direction = np.array([[0.2, -0.07], [-0.07, -0.1]])
    energy, potential = context.build_gamma(density)
    step = 1.0e-5
    e_plus, _ = context.build_gamma(density + step * direction)
    e_minus, _ = context.build_gamma(density - step * direction)
    finite_difference = (e_plus - e_minus) / (2.0 * step)

    assert np.isfinite(energy)
    assert finite_difference == pytest.approx(
        float(np.einsum("ij,ij->", potential, direction)),
        rel=2.0e-6,
        abs=2.0e-7,
    )
    assert provider.calls[-1]["periodic"] is True


def test_external_xc_adapter_rejects_unclosed_cell_domain():
    provider = _QuadraticDensityProvider()
    _, functional = _functional(provider)
    system, basis = _system_and_basis(length=4.0)
    lattice_options = _short_lattice_options(cutoff=4.1)
    cells = core.direct_lattice_cells(system, lattice_options.cutoff_bohr)

    with pytest.raises(ValueError, match="closed over active AO-image differences"):
        PeriodicExternalXC.prepare(
            basis,
            system,
            functional,
            grid_options=_coarse_atom_grid_options(),
            image_radius_bohr=4.1,
            lattice_options=lattice_options,
            cells=cells,
        )


def test_external_xc_domain_covers_boundary_crossing_multiatom_images():
    from vibeqc.pair_resolved_truncation import pair_resolved_domain

    provider = _QuadraticDensityProvider()
    _, functional = _functional(provider)
    length = 4.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * length,
        [
            vq.Atom(1, [0.1, 2.0, 2.0]),
            vq.Atom(1, [3.9, 2.0, 2.0]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lattice_options = _short_lattice_options(cutoff=3.0)
    context = PeriodicExternalXC.prepare(
        basis,
        system,
        functional,
        grid_options=_coarse_atom_grid_options(),
        image_radius_bohr=3.0,
        lattice_options=lattice_options,
    )

    active = pair_resolved_domain(system, 3.0).cells
    required = {
        tuple(
            (
                np.asarray(ket.index, dtype=int)
                - np.asarray(bra.index, dtype=int)
            ).tolist()
        )
        for bra in active
        for ket in active
    }
    old_radial = {
        tuple(np.asarray(cell.index, dtype=int).tolist())
        for cell in core.direct_lattice_cells(system, 6.0)
    }
    actual = {
        tuple(np.asarray(cell.index, dtype=int).tolist())
        for cell in context.cells
    }

    assert required.difference(old_radial)
    assert required.issubset(actual)


def test_external_xc_adapter_rejects_grid_profile_before_grid_build(
    monkeypatch,
):
    import vibeqc.periodic_external_xc as adapter_module

    provider = _QuadraticDensityProvider()
    _, functional = _functional(
        provider, required_grid_profile="pyscf-level3"
    )
    system, basis = _system_and_basis()

    def forbidden_grid_build(*args, **kwargs):
        raise AssertionError("profile mismatch must precede grid construction")

    monkeypatch.setattr(
        adapter_module, "build_periodic_becke_grid", forbidden_grid_build
    )
    with pytest.raises(ValueError, match="requires grid profile 'pyscf-level3'"):
        PeriodicExternalXC.prepare(
            basis,
            system,
            functional,
            grid_options=_coarse_atom_grid_options(),
            image_radius_bohr=2.0,
            lattice_options=_short_lattice_options(),
        )


def test_reduced_kmesh_detection_allows_identity_mapping_only():
    system, _ = _system_and_basis()
    native = core.monkhorst_pack(system, [2, 1, 1])
    full = SimpleNamespace(
        mesh=tuple(native.mesh),
        is_shift=tuple(native.is_shift),
        kpoints=list(native.kpoints),
        weights=list(native.weights),
        ir_mapping=[0, 1],
    )
    _reject_reduced_kmesh(full, system)

    reduced = SimpleNamespace(
        mesh=(2, 2, 1),
        is_shift=(0, 0, 0),
        kpoints=[np.zeros(3), np.ones(3)],
        weights=[0.5, 0.5],
        ir_mapping=[0, 0, 1, 1],
    )
    with pytest.raises(NotImplementedError, match="complete, uniformly weighted"):
        _reject_reduced_kmesh(reduced, system)


def test_explicit_weighted_kpoint_list_is_not_a_finite_torus():
    system, _ = _system_and_basis()
    explicit = SimpleNamespace(
        mesh=(1, 1, 1),
        is_shift=(0, 0, 0),
        kpoints=[np.zeros(3), np.array([0.2, 0.0, 0.0])],
        weights=[0.25, 0.75],
        ir_mapping=[],
    )
    with pytest.raises(NotImplementedError, match="complete, uniformly weighted"):
        _reject_reduced_kmesh(explicit, system)


@pytest.mark.parametrize("partition_radius", [2.0, 4.1, 6.0])
def test_external_xc_multik_weighted_complex_directional_derivative(partition_radius):
    provider = _QuadraticDensityProvider()
    _, functional = _functional(provider)
    system, basis = _system_and_basis(length=4.0)
    lattice_options = _short_lattice_options(cutoff=4.1)
    context = PeriodicExternalXC.prepare(
        basis,
        system,
        functional,
        grid_options=_coarse_atom_grid_options(),
        image_radius_bohr=partition_radius,
        lattice_options=lattice_options,
    )
    kmesh = core.monkhorst_pack(system, [2, 1, 1])

    density_0 = np.array(
        [[0.72, 0.11 + 0.04j], [0.11 - 0.04j, 0.48]], dtype=complex
    )
    density_k = [density_0, density_0.conj()]
    direction_0 = np.array(
        [[0.16, -0.05 + 0.09j], [-0.05 - 0.09j, -0.07]],
        dtype=complex,
    )
    direction_k = [direction_0, direction_0.conj()]

    _, potential_k = context.build_kpoints(density_k, kmesh)
    step = 1.0e-5
    e_plus, _ = context.build_kpoints(
        [D + step * H for D, H in zip(density_k, direction_k)], kmesh
    )
    e_minus, _ = context.build_kpoints(
        [D - step * H for D, H in zip(density_k, direction_k)], kmesh
    )
    finite_difference = (e_plus - e_minus) / (2.0 * step)
    weighted_trace = sum(
        float(weight)
        * float(np.real(np.trace(np.asarray(V) @ np.asarray(H))))
        for weight, V, H in zip(
            kmesh.weights, potential_k, direction_k
        )
    )

    assert finite_difference == pytest.approx(
        weighted_trace,
        rel=3.0e-6,
        abs=3.0e-7,
    )


def test_gpw_gamma_external_xc_bypasses_fft_xc(monkeypatch):
    import vibeqc.periodic_gapw_j as gpw_module

    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider)
    system, basis = _system_and_basis()
    pw_grid = PlaneWaveGrid(np.eye(3) * 8.0, 12, 12, 12)

    def forbidden(*args, **kwargs):
        raise AssertionError("pointwise FFT XC must not run for external XC")

    monkeypatch.setattr(gpw_module, "_evaluate_xc_on_grid", forbidden)
    result = gpw_module.run_periodic_rks_gpw(
        system,
        basis,
        functional=name,
        grid=pw_grid,
        max_iter=2,
        quiet=True,
        external_xc_grid_options=_coarse_atom_grid_options(),
        external_xc_image_radius_bohr=2.0,
        external_xc_lattice_options=_short_lattice_options(),
    )

    assert result.n_iter == 2
    assert np.isfinite(result.energy)
    assert np.isfinite(result.breakdown.e_xc)
    assert provider.calls
    build = result.fock_build
    assert build is not None
    assert build.source.provider == "gpw-grid+libint2+external-xc"
    assert result.energy == build.energy.total
    np.testing.assert_array_equal(result.density, build.channels[0].density)
    np.testing.assert_array_equal(result.fock, build.channels[0].operator.matrix)
    np.testing.assert_allclose(
        (result.mo_coeffs * result.occupations) @ result.mo_coeffs.T,
        result.density, atol=1e-10)
    assert dict(build.energy.components)["xc"] == result.breakdown.e_xc


@pytest.mark.parametrize("hf_fraction", [0.0, 0.25])
@pytest.mark.parametrize("max_iter", [0, 1, 30])
def test_gpw_external_evaluated_state(monkeypatch, max_iter, hf_fraction, record_property):
    import vibeqc.periodic_gapw_j as gpw_module
    from vibeqc.periodic_gpw_fock import _GpwRKSBuilder
    provider = _QuadraticDensityProvider()
    provider.provenance = {"model": "analytic-quadratic-density", "revision": 1}
    name, _ = _functional(provider, hf_exchange_fraction=hf_fraction)
    system, basis = _system_and_basis()
    builders = []
    original = _GpwRKSBuilder.__call__

    def capture(self, density):
        if not builders:
            builders.append(self)
        return original(self, density)

    monkeypatch.setattr(_GpwRKSBuilder, "__call__", capture)
    result = gpw_module.run_periodic_rks_gpw(
        system, basis, functional=name,
        grid=PlaneWaveGrid(np.eye(3) * 8.0, 12, 12, 12),
        max_iter=max_iter, quiet=True, use_diis=False,
        external_xc_grid_options=_coarse_atom_grid_options(),
        external_xc_image_radius_bohr=2.0,
        external_xc_lattice_options=_short_lattice_options())
    builder = builders[0]
    d = result.density.copy()
    direction = np.array([[.13, -.07], [-.07, .04]])
    step = 1e-5
    plus, minus = builder(d + step*direction), builder(d - step*direction)
    derivative = (plus.energy.total-minus.energy.total)/(2*step)
    assert derivative == pytest.approx(np.sum(result.fock*direction), abs=2e-7)
    record_property("energy_Eh", result.energy)
    record_property("derivative_error", abs(derivative-np.sum(result.fock*direction)))
    from vibeqc.periodic_gapw_runner_adapter import gpw_result_to_runner_shape
    from vibeqc.output.formats.qvf import qvf_wf_data
    adapted = gpw_result_to_runner_shape(result, system, basis)
    assert adapted.fock_build is result.fock_build
    assert adapted.run_fields == result.run_fields
    wf = qvf_wf_data(adapted, basis, system.unit_cell_molecule())
    assert wf["mo_metadata"]["orbital_kind"] == "natural"
    np.testing.assert_allclose(
        (adapted.mo_coeffs*adapted.occupations) @ adapted.mo_coeffs.T, d, atol=1e-10)
    source = result.fock_build.source.as_dict()["details"]["xc"]["details"]
    assert source["checkpoint"] == provider.provenance
    assert source["hf_exchange_fraction"] == hf_fraction
    provider.provenance["revision"] = 2
    with pytest.raises(ValueError, match="source changed"):
        builder(d)
    provider.provenance["revision"] = 1
    assert result.fock_build.energy.total == result.energy
    np.testing.assert_allclose((result.mo_coeffs*result.occupations) @ result.mo_coeffs.T,
                               d, atol=1e-10)
    if max_iter == 30:
        assert result.converged
    # Neither a writable result copy nor later builds alter the captured state.
    result.density[:] = 0
    np.testing.assert_array_equal(result.fock_build.channels[0].density, d)
    builder.external_xc.lattice_options.cutoff_bohr += .1
    with pytest.raises(ValueError, match="source changed"):
        builder(d)


def test_gpw_multik_external_xc_bypasses_all_fft_xc(monkeypatch):
    import vibeqc.periodic_gapw_j as gpw_module
    from vibeqc.dft_plus_u import ao_group_indices
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider)
    system, basis = _system_and_basis()
    pw_grid = PlaneWaveGrid(np.eye(3) * 8.0, 10, 10, 10)
    kmesh = core.monkhorst_pack(system, [2, 1, 1])
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)]

    def forbidden(*args, **kwargs):
        raise AssertionError("FFT XC must not run for external XC")

    monkeypatch.setattr(gpw_module, "_evaluate_xc_on_grid", forbidden)
    monkeypatch.setattr(gpw_module, "_xc_effective_potential_grid", forbidden)
    result = gpw_module.run_periodic_rks_gpw_multi_k(
        system,
        basis,
        kmesh,
        functional=name,
        grid=pw_grid,
        max_iter=1,
        quiet=True,
        dft_plus_u_sites=sites,
        external_xc_grid_options=_coarse_atom_grid_options(),
        external_xc_image_radius_bohr=2.0,
        external_xc_lattice_options=_short_lattice_options(),
    )

    assert result.n_iter == 1
    assert len(result.mo_coeffs_k) == 2
    assert np.isfinite(result.energy)
    assert provider.calls

    # On a deliberate max_iter=1 exit the accepted density is authoritative;
    # final orbitals are canonical eigenvectors of F[D] and need not rebuild
    # that non-self-consistent density until another SCF iteration is taken.
    density_k = [np.asarray(block) for block in result.density_k]
    independent_xc = PeriodicExternalXC.prepare(
        basis,
        system,
        core.Functional(name, 1),
        grid_options=_coarse_atom_grid_options(),
        image_radius_bohr=2.0,
        lattice_options=_short_lattice_options(),
    )
    expected_xc, expected_vxc_k = independent_xc.build_kpoints(
        density_k, kmesh
    )
    assert result.breakdown.e_xc == pytest.approx(expected_xc, abs=1.0e-12)
    assert len(result.fock_k) == len(kmesh.kpoints)
    assert len(result.overlap_k) == len(kmesh.kpoints)
    assert len(result.hcore_k) == len(kmesh.kpoints)

    lattice_options = gpw_module.multik_one_electron_lattice_options(
        basis,
        system,
    )
    overlap_lattice = core.compute_overlap_lattice(
        basis,
        system,
        lattice_options,
    )
    kinetic_lattice = core.compute_kinetic_lattice(
        basis,
        system,
        lattice_options,
    )
    nuclear_lattice = compute_nuclear_lattice_dispatch(
        basis,
        system,
        gpw_module.multik_v_ne_lattice_options(basis, system),
    )
    overlap_k = [
        np.asarray(core.bloch_sum(overlap_lattice, point), dtype=complex)
        for point in kmesh.kpoints
    ]
    overlap_k = [0.5 * (block + block.conj().T) for block in overlap_k]
    hcore_k = []
    for point in kmesh.kpoints:
        kinetic = np.asarray(
            core.bloch_sum(kinetic_lattice, point), dtype=complex
        )
        nuclear = np.asarray(
            core.bloch_sum(nuclear_lattice, point), dtype=complex
        )
        block = kinetic + nuclear
        hcore_k.append(0.5 * (block + block.conj().T))

    assert gpw_module._multik_gpw_is_molecular_limit(system) is False
    grid_points = pw_grid.cartesian_coords().reshape(-1, 3)
    direct_cells = core.direct_lattice_cells(system, 25.0)
    translations = np.asarray(
        [cell.r_cart for cell in direct_cells], dtype=float
    )
    bloch_ao_k = [
        gpw_module.bloch_ao_on_grid(
            basis,
            grid_points,
            point,
            translations,
        )
        for point in kmesh.kpoints
    ]
    rho_grid = gpw_module.collocate_bloch_density_on_grid(
        bloch_ao_k,
        density_k,
        kmesh.weights,
        pw_grid,
    )
    hartree_grid = core.solve_poisson_coulomb(
        rho_grid,
        pw_grid.lattice_bohr,
    )
    expected_j_k = [
        gpw_module.project_potential_to_bloch_ao(
            bloch_ao,
            hartree_grid,
            pw_grid,
        )
        for bloch_ao in bloch_ao_k
    ]
    for ik in range(len(kmesh.kpoints)):
        np.testing.assert_allclose(
            result.overlap_k[ik], overlap_k[ik], atol=1.0e-13
        )
        np.testing.assert_allclose(
            result.hcore_k[ik], hcore_k[ik], atol=1.0e-13
        )
        np.testing.assert_allclose(
            result.fock_k[ik], result.fock_k[ik].conj().T, atol=1.0e-13
        )
        _assert_generalized_eigenpairs(
            result.fock_k[ik],
            result.overlap_k[ik],
            result.mo_coeffs_k[ik],
            result.mo_energies_k[ik],
        )
    groups = ao_group_indices(basis)
    sites_cxx = [
        core._HubbardSiteCxx(
            site.atom_index,
            site.l,
            site.U_eff_hartree,
        )
        for site in sites
    ]
    e_sigma, v_ao = core._compute_dft_plus_u_multi_k_per_spin_cxx(
        sites_cxx,
        [groups[(site.atom_index, site.l)] for site in sites],
        overlap_k,
        [0.5 * density for density in density_k],
        list(kmesh.weights),
    )
    hartree_from_operator = 0.0
    v_ao = np.asarray(v_ao, dtype=complex)
    for weight, density, fock, overlap, hcore, vxc, expected_j in zip(
        kmesh.weights,
        density_k,
        result.fock_k,
        result.overlap_k,
        result.hcore_k,
        expected_vxc_k,
        expected_j_k,
        strict=True,
    ):
        v_u = overlap @ v_ao @ overlap
        j_matrix = fock - hcore - np.asarray(vxc) - v_u
        np.testing.assert_allclose(
            j_matrix,
            expected_j,
            atol=5.0e-10,
        )
        hartree_from_operator += float(weight) * float(
            np.real(np.einsum("ij,ji->", density, j_matrix))
        )
    assert result.breakdown.e_hartree == pytest.approx(
        0.5 * hartree_from_operator,
        abs=5.0e-10,
    )
    assert result.e_dft_plus_u == pytest.approx(
        2.0 * float(e_sigma),
        abs=1.0e-12,
    )
    assert result.energy == pytest.approx(
        result.breakdown.e_total,
        abs=1.0e-12,
    )


def test_gapw_gamma_external_xc_bypasses_split_xc(monkeypatch):
    import vibeqc.periodic_gapw_augment as gapw_module
    from vibeqc.dft_plus_u import (
        ao_group_indices,
        compute_dudarev_energy,
        compute_occupation_matrices,
    )
    from vibeqc.periodic_gapw_j import _overlap_lattice_gamma

    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider)
    length = 8.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * length,
        [vq.Atom(8, [length / 2, length / 2, length / 2])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    pw_grid = PlaneWaveGrid(np.eye(3) * length, 10, 10, 10)
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)]

    def forbidden(*args, **kwargs):
        raise AssertionError("GAPW split XC must not run for external XC")

    monkeypatch.setattr(
        gapw_module.GapwJBuilder, "_smooth_xc_generation", forbidden
    )
    monkeypatch.setattr(
        gapw_module.GapwJBuilder, "_analytic_xc_correction", forbidden
    )
    monkeypatch.setattr(
        gapw_module.GapwAugmentation, "compute_xc_correction", forbidden
    )
    original_augmentation_energy = (
        gapw_module.GapwAugmentation.compute_augmentation_energy
    )

    def hartree_only_augmentation(self, density, functional=None):
        assert functional is None
        return original_augmentation_energy(
            self, density, functional=functional
        )

    monkeypatch.setattr(
        gapw_module.GapwAugmentation,
        "compute_augmentation_energy",
        hartree_only_augmentation,
    )

    result = gapw_module.run_periodic_rks_gapw(
        system,
        basis,
        functional=name,
        grid=pw_grid,
        max_iter=1,
        quiet=True,
        one_centre="block",
        n_radial=4,
        lebedev_order=3,
        dft_plus_u_sites=sites,
        external_xc_grid_options=_coarse_atom_grid_options(),
        external_xc_image_radius_bohr=2.0,
        external_xc_lattice_options=_short_lattice_options(),
    )

    assert result.n_iter == 1
    assert np.isfinite(result.energy)
    assert np.isfinite(result.breakdown.e_xc)
    assert provider.calls
    _assert_generalized_eigenpairs(
        result.fock,
        result.overlap,
        result.mo_coeffs,
        result.mo_energies,
    )

    overlap = _overlap_lattice_gamma(basis, system)
    occupation = compute_occupation_matrices(
        sites,
        0.5 * np.asarray(result.density),
        overlap,
        ao_group_indices(basis),
    )
    expected_u = 2.0 * compute_dudarev_energy(sites, occupation)
    assert result.e_dft_plus_u == pytest.approx(expected_u, abs=1.0e-12)
    assert result.breakdown.e_dft_plus_u == pytest.approx(
        expected_u,
        abs=1.0e-12,
    )


@pytest.mark.parametrize(
    "request_shape",
    ("plain", "positive-smearing", "charged", "ecp-paired"),
)
def test_direct_gapw_multik_external_xc_fails_before_setup(
    monkeypatch,
    request_shape,
):
    import vibeqc.periodic_gapw_augment as gapw_module
    import vibeqc.periodic_gapw_grid as grid_module

    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider)
    system, basis = _system_and_basis()
    kwargs = {}
    if request_shape == "positive-smearing":
        kwargs["smearing_temperature"] = 0.01
    elif request_shape == "charged":
        system = vq.PeriodicSystem(
            3,
            np.eye(3) * 8.0,
            [
                vq.Atom(1, [4.0, 4.0, 3.3]),
                vq.Atom(1, [4.0, 4.0, 4.7]),
            ],
            charge=2,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    elif request_shape == "ecp-paired":
        system, basis = _ecp_bearing_system_and_basis()

    kmesh = core.monkhorst_pack(system, [2, 1, 1])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("multi-k GAPW external-XC guard ran after setup")

    monkeypatch.setattr(grid_module, "make_grid", forbidden)
    monkeypatch.setattr(core, "compute_kinetic_lattice", forbidden)
    with pytest.raises(NotImplementedError, match="GAPW at Gamma only"):
        gapw_module.run_periodic_rks_gapw_multi_k(
            system,
            basis,
            kmesh,
            functional=name,
            quiet=True,
            **kwargs,
        )

    assert provider.calls == []


@pytest.mark.parametrize(
    ("request_shape", "message"),
    (
        ("non-3d", "only for a 3D periodic GPW Hamiltonian"),
        ("charged", "charged-cell external XC"),
        ("ecp-paired", "ECP-bearing bases"),
    ),
)
def test_external_evaluate_gpw_energy_rejects_before_setup(
    monkeypatch,
    request_shape,
    message,
):
    import vibeqc.periodic_gapw_grid as grid_module
    import vibeqc.periodic_gapw_j as gpw_module

    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider)
    system, basis = _system_and_basis()
    if request_shape == "non-3d":
        system = vq.PeriodicSystem(
            2,
            np.eye(3) * 8.0,
            [
                vq.Atom(1, [4.0, 4.0, 3.3]),
                vq.Atom(1, [4.0, 4.0, 4.7]),
            ],
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    elif request_shape == "charged":
        system = vq.PeriodicSystem(
            3,
            np.eye(3) * 8.0,
            [
                vq.Atom(1, [4.0, 4.0, 3.3]),
                vq.Atom(1, [4.0, 4.0, 4.7]),
            ],
            charge=2,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    elif request_shape == "ecp-paired":
        system, basis = _ecp_bearing_system_and_basis()
    density = np.zeros((basis.nbasis, basis.nbasis))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("evaluate_gpw_energy external guard ran after setup")

    monkeypatch.setattr(grid_module, "make_grid", forbidden)
    monkeypatch.setattr(gpw_module, "_kinetic_lattice_gamma", forbidden)
    with pytest.raises(NotImplementedError, match=message):
        gpw_module.evaluate_gpw_energy(
            system,
            basis,
            density,
            functional=name,
            quiet=True,
        )

    assert provider.calls == []


@pytest.mark.parametrize("route", ("gpw-gamma", "gpw-multik", "gapw-gamma"))
def test_external_gpw_gapw_reject_positive_smearing_before_integrals(
    monkeypatch,
    route,
):
    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider)
    system, basis = _system_and_basis()
    pw_grid = PlaneWaveGrid(np.eye(3) * 8.0, 8, 8, 8)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("external-XC smearing guard ran after setup")

    if route == "gpw-gamma":
        import vibeqc.periodic_gapw_j as module

        monkeypatch.setattr(module, "_kinetic_lattice_gamma", forbidden)
        def call():
            return module.run_periodic_rks_gpw(
                system,
                basis,
                functional=name,
                grid=pw_grid,
                smearing_temperature=0.01,
                quiet=True,
            )
    elif route == "gpw-multik":
        import vibeqc.periodic_gapw_j as module

        monkeypatch.setattr(core, "compute_kinetic_lattice", forbidden)
        def call():
            return module.run_periodic_rks_gpw_multi_k(
                system,
                basis,
                core.monkhorst_pack(system, [2, 1, 1]),
                functional=name,
                grid=pw_grid,
                smearing_temperature=0.01,
                quiet=True,
            )
    else:
        import vibeqc.periodic_gapw_augment as module

        monkeypatch.setattr(module, "_kinetic_lattice_gamma", forbidden)
        def call():
            return module.run_periodic_rks_gapw(
                system,
                basis,
                functional=name,
                grid=pw_grid,
                smearing_temperature=0.01,
                one_centre="block",
                quiet=True,
            )

    with pytest.raises(NotImplementedError, match="zero-temperature only"):
        call()

    assert provider.calls == []


@pytest.mark.parametrize("jk_method", ("gpw", "gapw"))
def test_high_level_external_gpw_gapw_reject_charged_cell_before_dry_run(
    tmp_path,
    jk_method,
):
    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider)
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / f"charged-external-{jk_method}"

    with pytest.raises(NotImplementedError, match="charged-cell GPW/GAPW"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional=name,
            jk_method=jk_method,
            output=stem,
            dry_run=True,
            progress=False,
        )

    assert_preflight_refusal(stem)


@pytest.mark.parametrize("route", ("gpw-gamma", "gpw-multik", "gapw-gamma"))
def test_low_level_external_gpw_gapw_reject_charged_cell(route):
    import vibeqc.periodic_gapw_augment as gapw_module
    import vibeqc.periodic_gapw_j as gpw_module

    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider)
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(NotImplementedError, match="charged-cell external XC"):
        if route == "gpw-gamma":
            gpw_module.run_periodic_rhf_gpw(
                system,
                basis,
                functional=name,
                quiet=True,
            )
        elif route == "gpw-multik":
            gpw_module.run_periodic_rks_gpw_multi_k(
                system,
                basis,
                core.monkhorst_pack(system, [2, 1, 1]),
                functional=name,
                quiet=True,
            )
        else:
            gapw_module.run_periodic_rhf_gapw(
                system,
                basis,
                functional=name,
                quiet=True,
            )


@pytest.mark.parametrize("route", ("gpw-gamma", "gpw-multik", "gapw-gamma"))
def test_low_level_external_gpw_gapw_reject_ecp_paired_basis(route):
    import vibeqc.periodic_gapw_augment as gapw_module
    import vibeqc.periodic_gapw_j as gpw_module

    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider)
    system, basis = _ecp_bearing_system_and_basis()

    with pytest.raises(NotImplementedError, match="ECP-bearing bases"):
        if route == "gpw-gamma":
            gpw_module.run_periodic_rhf_gpw(
                system,
                basis,
                functional=name,
                quiet=True,
            )
        elif route == "gpw-multik":
            gpw_module.run_periodic_rks_gpw_multi_k(
                system,
                basis,
                core.monkhorst_pack(system, [2, 1, 1]),
                functional=name,
                quiet=True,
            )
        else:
            gapw_module.run_periodic_rhf_gapw(
                system,
                basis,
                functional=name,
                quiet=True,
            )


@pytest.mark.parametrize("mode", ["gamma", "uks", "multik"])
@pytest.mark.parametrize("requested_reach", [0.0, -1.0, 2.0, 7.0])
def test_external_xc_partition_does_not_change_density_support(
    monkeypatch, mode, requested_reach,
):
    """Hold quadrature fixed to isolate the AO support from partition error."""
    provider = _QuadraticDensityProvider()
    _, functional = _functional(provider)
    system, basis = _system_and_basis(length=4.0)
    options = _short_lattice_options(cutoff=4.1)
    options.becke_image_radius_bohr = requested_reach
    grid = external_xc.build_periodic_becke_grid(
        system, grid_options=_coarse_atom_grid_options(), image_radius_bohr=4.1,
    )
    radii = []

    def fixed_grid(*args, **kwargs):
        radii.append(kwargs["image_radius_bohr"])
        return grid

    monkeypatch.setattr(external_xc, "build_periodic_becke_grid", fixed_grid)
    expected_reach = min(requested_reach, 4.1) if requested_reach > 0 else 4.1
    density = np.array([[0.7, 0.1], [0.1, 0.5]])
    mesh = core.monkhorst_pack(system, [2, 1, 1])
    outputs, inputs, domains = [], [], []
    for partition in (2.0, 4.1, 6.0):
        context = PeriodicExternalXC.prepare(
            basis, system, functional, image_radius_bohr=partition,
            lattice_options=options,
        )
        assert context.lattice_options.becke_image_radius_bohr == expected_reach
        domains.append([tuple(c.index) for c in context.cells])
        if mode == "gamma":
            outputs.append(context.build_gamma(density))
        elif mode == "uks":
            outputs.append(context.build_gamma_uks(0.6 * density, 0.4 * density))
        else:
            outputs.append(context.build_kpoints([density, density], mesh))
        inputs.append({key: np.asarray(provider.calls[-1][key]).copy()
                       for key in ("rho_alpha", "rho_beta", "tau_alpha", "tau_beta")})
    assert radii == [2.0, 4.1, 6.0]
    assert options.becke_image_radius_bohr == requested_reach
    assert domains[0] == domains[1] == domains[2]
    for actual, features in zip(outputs[1:], inputs[1:]):
        for reference, value in zip(outputs[0], actual):
            np.testing.assert_array_equal(value, reference)
        for key in features:
            np.testing.assert_array_equal(features[key], inputs[0][key])


@pytest.mark.parametrize("field,value", [
    ("cutoff_bohr", 0.0), ("cutoff_bohr", -1.0),
    ("cutoff_bohr", np.nan), ("cutoff_bohr", np.inf),
    ("becke_image_radius_bohr", np.nan), ("becke_image_radius_bohr", np.inf),
])
def test_external_xc_rejects_invalid_density_support_before_grid(monkeypatch, field, value):
    _, functional = _functional(_QuadraticDensityProvider())
    system, basis = _system_and_basis()
    options = _short_lattice_options()
    setattr(options, field, value)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid density support reached grid construction")

    monkeypatch.setattr(external_xc, "build_periodic_becke_grid", unexpected)
    with pytest.raises(ValueError, match="must be finite"):
        PeriodicExternalXC.prepare(basis, system, functional, lattice_options=options)


@pytest.mark.parametrize("hf_fraction", [0.0, 0.25])
def test_gpw_external_scf_preserves_density_reach(monkeypatch, hf_fraction, record_property):
    """Converged analytic-provider regression, not a SKALA checkpoint test."""
    from vibeqc.periodic_gapw_j import run_periodic_rks_gpw

    provider = _QuadraticDensityProvider()
    name, _ = _functional(provider, hf_exchange_fraction=hf_fraction)
    system, basis = _system_and_basis(length=4.0)
    grid_options = _coarse_atom_grid_options()
    atom_grid = external_xc.build_periodic_becke_grid(
        system, grid_options=grid_options, image_radius_bohr=4.1,
    )
    monkeypatch.setattr(external_xc, "build_periodic_becke_grid", lambda *a, **kw: atom_grid)
    energies = []
    for partition, density_reach in ((2.0, 0.0), (4.1, 0.0), (2.0, 2.0)):
        options = _short_lattice_options(cutoff=4.1)
        options.becke_image_radius_bohr = density_reach
        result = run_periodic_rks_gpw(
            system, basis, functional=name,
            grid=PlaneWaveGrid(np.eye(3) * 4.0, 12, 12, 12),
            max_iter=60, quiet=True, use_diis=False,
            external_xc_grid_options=grid_options,
            external_xc_image_radius_bohr=partition,
            external_xc_lattice_options=options,
        )
        assert result.converged
        energies.append(result.energy)
    # The first two runs have the same density operator and fixed quadrature.
    # Independent SCFs can differ by a rounding step in parallel reductions.
    # Keep the comparison at machine roundoff, far below SCF convergence.
    assert energies[0] == pytest.approx(energies[1], rel=0., abs=32*np.finfo(float).eps)
    # Explicitly requesting the old truncated support remains possible and
    # actually changes this cell: it must not be selected by partition alone.
    assert abs(energies[2] - energies[0]) > 1.0e-4
    record_property("full_reach_energy_Eh", energies[0])
    record_property("narrow_reach_energy_Eh", energies[2])
    record_property("partition_only_energy_difference_Eh", energies[1] - energies[0])
