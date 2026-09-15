"""KS-DFT on a Cyclic Cluster Model reference (RKS-CCM / UKS-CCM, 4-center).

The KS-CCM Fock is ``h^CCM + J^CCM + V_xc (− α K^CCM)`` — the four-center WSSC
Coulomb/exact-exchange (``method="union12"`` or ``"aiccm2026dev-a"``) plus a semi-local
XC potential on a supercell Becke grid. These tests pin the molecular limit: an
isolated cluster must reproduce vibe-qc's molecular ``run_rks`` / ``run_uks`` for
pure and hybrid functionals (the CCM Coulomb is the molecular Coulomb and the XC
grid is the molecular grid there).

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import (
    Atom,
    BasisSet,
    PeriodicSystem,
    RKSOptions,
    UKSOptions,
    run_rks,
    run_uks,
)
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.dft import (
    _four_center_external_xc_context,
    run_ccm_rks,
    run_ccm_uks,
)

pytestmark = pytest.mark.experimental  # literal four-centre WSSC research lane

BOHR = 1.0 / 0.529177210903
_EXTERNAL_NAMES = itertools.count()


class _QuadraticFullGridProvider:
    """Small variational provider: E = scale/2 (integral rho)^2."""

    def __init__(self, scale=0.02):
        self.scale = float(scale)
        self.calls = []

    def __call__(self, features):
        self.calls.append(features)
        weights = np.asarray(features["grid_weights"], dtype=float)
        rho_alpha = np.asarray(features["rho_alpha"], dtype=float)
        rho_beta = np.asarray(features["rho_beta"], dtype=float)
        integral = float(weights @ (rho_alpha + rho_beta))
        q = self.scale * integral * weights
        zeros = np.zeros(weights.size)
        zeros_gradient = np.zeros((weights.size, 3))
        return {
            "energy": 0.5 * self.scale * integral * integral,
            "v_rho_alpha": q,
            "v_rho_beta": q,
            "v_grad_alpha": zeros_gradient,
            "v_grad_beta": zeros_gradient,
            "v_tau_alpha": zeros,
            "v_tau_beta": zeros,
        }


def _external_functional(
    provider,
    *,
    hf_exchange_fraction=0.0,
    required_grid_profile="",
):
    name = f"test-fourcenter-full-grid-{next(_EXTERNAL_NAMES)}"
    vq.define_external_functional(
        name,
        provider,
        hf_exchange_fraction=hf_exchange_fraction,
        required_grid_profile=required_grid_profile,
    )
    return name


def _coarse_grid():
    options = vq.GridOptions()
    options.n_radial = 4
    options.n_theta = 3
    options.n_phi = 4
    return options


def _short_xc_lattice(cutoff=6.1):
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = float(cutoff)
    options.nuclear_cutoff_bohr = float(cutoff)
    return options


def _compact_h2_ccm():
    cell = PeriodicSystem(
        3,
        np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])],
        0,
        1,
    )
    return CCMSystem(cell, (2, 1, 1), "sto-3g")


def _assert_generalized_fock_eigenpairs(fock, overlap, coeffs, energies):
    fock = np.asarray(fock, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    energies = np.asarray(energies, dtype=float)
    residual = fock @ coeffs - (overlap @ coeffs) * energies[None, :]
    assert np.max(np.abs(residual)) < 5.0e-10
    np.testing.assert_allclose(
        coeffs.T @ overlap @ coeffs,
        np.eye(coeffs.shape[1]),
        atol=5.0e-10,
        rtol=5.0e-10,
    )


def _h2():
    return PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)


@pytest.mark.parametrize("functional", ["pbe", "pbe0", "b3lyp"])
@pytest.mark.parametrize("method", ["union12", "aiccm2026dev-a"])
def test_rks_ccm_isolated_equals_molecular(functional, method):
    """Isolated H2: RKS-CCM == molecular run_rks (pure + hybrid functionals)."""
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    r = run_ccm_rks(ccm, functional, method=method)
    mol = ccm.supercell
    opts = RKSOptions()
    opts.functional = functional
    ref = run_rks(mol, BasisSet(mol, "sto-3g"), opts)
    assert r.converged
    assert r.energy == pytest.approx(ref.energy, abs=1e-6)


def test_ks_result_backend_identity_and_hold_flag():
    """IID 344: the KS-CCM results identify their executing route/operator
    (``backend``) and positively report the parity-hold state. The
    four-center KS cluster routes never touch the multi-k GDF hold class,
    so held=False is an affirmative verdict a record consumer can act on."""
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    r = run_ccm_rks(ccm, "pbe", method="union12")
    assert r.backend == "ccm-fourcenter-direct-union12-rks"
    assert r.parity_held is False
    u = run_ccm_uks(ccm, "pbe", method="union12")
    assert u.backend == "ccm-fourcenter-direct-union12-uks"
    assert u.parity_held is False


def test_four_center_external_hybrid_fails_before_integrals(monkeypatch):
    """Only pure external XC is enabled on the literal WSSC operator."""
    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider, hf_exchange_fraction=0.25)
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")

    def forbidden_integrals(*args, **kwargs):
        raise AssertionError("external hybrid gate must precede CCM integrals")

    import vibeqc.periodic.ccm.dft as dft_module

    monkeypatch.setattr(dft_module, "ccm_overlap", forbidden_integrals)
    for driver in (run_ccm_rks, run_ccm_uks):
        with pytest.raises(NotImplementedError, match="pure full-grid"):
            driver(ccm, name, method="aiccm2026dev-a")
    assert provider.calls == []


def test_four_center_external_grid_profile_fails_before_integrals(monkeypatch):
    provider = _QuadraticFullGridProvider()
    name = _external_functional(
        provider, required_grid_profile="pyscf-level3"
    )
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")

    def forbidden_integrals(*args, **kwargs):
        raise AssertionError("grid profile gate must precede CCM integrals")

    import vibeqc.periodic.ccm.dft as dft_module

    monkeypatch.setattr(dft_module, "ccm_overlap", forbidden_integrals)
    with pytest.raises(ValueError, match="requires grid profile 'pyscf-level3'"):
        run_ccm_rks(
            ccm,
            name,
            method="aiccm2026dev-a",
            grid_options=_coarse_grid(),
            becke_image_radius_bohr=2.0,
            xc_lattice_options=_short_xc_lattice(2.0),
        )
    assert provider.calls == []


def test_four_center_external_rks_grid_is_periodic_and_variational():
    """Janetzko Eqs. 27-32: reference-cluster grid and periodic AO images."""
    ccm = _compact_h2_ccm()
    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider)
    context = _four_center_external_xc_context(
        ccm,
        name,
        1,
        grid_options=_coarse_grid(),
        becke_image_radius_bohr=6.1,
        xc_lattice_options=_short_xc_lattice(),
        who="test_four_center_external_rks_grid_is_periodic_and_variational",
    )
    nbf = int(ccm.basis.nbasis)
    density = 0.27 * np.eye(nbf)
    direction = np.arange(1, nbf * nbf + 1, dtype=float).reshape(nbf, nbf)
    direction = 0.5 * (direction + direction.T)
    direction /= np.linalg.norm(direction)

    _, potential = context.build_gamma(density)
    step = 2.0e-7
    e_plus, _ = context.build_gamma(density + step * direction)
    e_minus, _ = context.build_gamma(density - step * direction)
    finite_difference = (e_plus - e_minus) / (2.0 * step)
    assert float(np.sum(potential * direction)) == pytest.approx(
        finite_difference, rel=3.0e-6, abs=3.0e-7
    )

    payload = provider.calls[0]
    assert payload["periodic"] is True
    assert payload["periodic_dimension"] == 3
    np.testing.assert_allclose(payload["lattice"], ccm.cluster_system.lattice)
    assert len(payload["atomic_grid_sizes"]) == len(ccm.supercell.atoms)
    indices = {
        tuple(int(value) for value in np.asarray(cell.index))
        for cell in context.cells
    }
    assert (0, 1, 0) in indices and (0, -1, 0) in indices


def test_four_center_external_uks_is_variational_and_matches_equal_spin_rks():
    ccm = _compact_h2_ccm()
    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider)
    common = dict(
        grid_options=_coarse_grid(),
        becke_image_radius_bohr=6.1,
        xc_lattice_options=_short_xc_lattice(),
    )
    rks = _four_center_external_xc_context(
        ccm, name, 1, who="test-fourcenter-rks", **common
    )
    uks = _four_center_external_xc_context(
        ccm, name, 2, who="test-fourcenter-uks", **common
    )
    nbf = int(ccm.basis.nbasis)
    density = 0.31 * np.eye(nbf)
    e_rks, v_rks = rks.build_gamma(density)
    e_uks, v_alpha, v_beta = uks.build_gamma_uks(
        0.5 * density, 0.5 * density
    )
    assert e_uks == pytest.approx(e_rks, abs=2.0e-11)
    np.testing.assert_allclose(v_alpha, v_rks, atol=2.0e-10, rtol=2.0e-10)
    np.testing.assert_allclose(v_beta, v_rks, atol=2.0e-10, rtol=2.0e-10)

    direction_alpha = np.arange(
        1, nbf * nbf + 1, dtype=float
    ).reshape(nbf, nbf)
    direction_alpha = 0.5 * (direction_alpha + direction_alpha.T)
    direction_beta = np.flipud(direction_alpha).copy()
    direction_beta = 0.5 * (direction_beta + direction_beta.T)
    norm = np.sqrt(
        np.sum(direction_alpha**2) + np.sum(direction_beta**2)
    )
    direction_alpha /= norm
    direction_beta /= norm
    density_alpha = 0.19 * np.eye(nbf)
    density_beta = 0.12 * np.eye(nbf)
    _, va, vb = uks.build_gamma_uks(density_alpha, density_beta)
    step = 2.0e-7
    e_plus, _, _ = uks.build_gamma_uks(
        density_alpha + step * direction_alpha,
        density_beta + step * direction_beta,
    )
    e_minus, _, _ = uks.build_gamma_uks(
        density_alpha - step * direction_alpha,
        density_beta - step * direction_beta,
    )
    finite_difference = (e_plus - e_minus) / (2.0 * step)
    projected = float(
        np.sum(va * direction_alpha) + np.sum(vb * direction_beta)
    )
    assert projected == pytest.approx(
        finite_difference, rel=3.0e-6, abs=3.0e-7
    )


def test_four_center_external_rks_and_equal_spin_uks_run_end_to_end():
    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider)
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    common = dict(
        method="aiccm2026dev-a",
        grid_options=_coarse_grid(),
        becke_image_radius_bohr=2.0,
        xc_lattice_options=_short_xc_lattice(2.0),
    )
    restricted = run_ccm_rks(ccm, name, **common)
    unrestricted = run_ccm_uks(ccm, name, **common)
    assert restricted.converged and unrestricted.converged
    assert unrestricted.energy == pytest.approx(restricted.energy, abs=1.0e-8)
    np.testing.assert_allclose(
        unrestricted.density, restricted.density, atol=2.0e-8, rtol=2.0e-8
    )
    assert restricted.backend == "ccm-fourcenter-direct-aiccm2026dev-a-rks"
    assert unrestricted.backend == "ccm-fourcenter-direct-aiccm2026dev-a-uks"


def test_four_center_external_rks_orbitals_match_returned_fock():
    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider)
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    result = run_ccm_rks(
        ccm,
        name,
        method="aiccm2026dev-a",
        grid_options=_coarse_grid(),
        becke_image_radius_bohr=2.0,
        xc_lattice_options=_short_xc_lattice(2.0),
    )
    assert result.converged
    _assert_generalized_fock_eigenpairs(
        result.fock,
        result.overlap,
        result.mo_coeffs,
        result.mo_energies,
    )


def test_four_center_external_uks_spin_orbitals_match_returned_focks():
    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider)
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    result = run_ccm_uks(
        ccm,
        name,
        method="aiccm2026dev-a",
        grid_options=_coarse_grid(),
        becke_image_radius_bohr=2.0,
        xc_lattice_options=_short_xc_lattice(2.0),
    )
    assert result.converged
    assert result.fock_beta is not None
    _assert_generalized_fock_eigenpairs(
        result.fock,
        result.overlap,
        result.mo_coeffs,
        result.mo_energies,
    )
    _assert_generalized_fock_eigenpairs(
        result.fock_beta,
        result.overlap,
        result.mo_coeffs_beta,
        result.mo_energies_beta,
    )


@pytest.mark.parametrize(
    "driver_name",
    ("wssc-rks", "wssc-uks", "real-gamma-rks", "real-gamma-uks"),
)
def test_external_ccm_low_level_rejects_ecp_paired_basis(driver_name):
    from vibeqc.periodic.ccm.direct import (
        run_ccm_rks_direct,
        run_ccm_uks_direct,
    )

    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider)
    # H/LANL2DZ is all-electron; test an element whose core is replaced.
    unit = PeriodicSystem(3, np.eye(3) * 8.0, [Atom(30, [0, 0, 0])], 0, 1)
    ccm = CCMSystem(unit, (1, 1, 1), "lanl2dz")
    drivers = {
        "wssc-rks": run_ccm_rks,
        "wssc-uks": run_ccm_uks,
        "real-gamma-rks": run_ccm_rks_direct,
        "real-gamma-uks": run_ccm_uks_direct,
    }

    with pytest.raises(NotImplementedError, match="ECP-bearing bases"):
        drivers[driver_name](ccm, name, max_iter=1)


@pytest.mark.parametrize("spin", (1, 2))
def test_external_real_gamma_low_level_rejects_charged_cell(spin):
    from vibeqc.periodic.ccm.direct import (
        run_ccm_rks_direct,
        run_ccm_uks_direct,
    )

    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider)
    cell = PeriodicSystem(
        3,
        np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])],
        2,
        1,
    )
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    driver = run_ccm_rks_direct if spin == 1 else run_ccm_uks_direct

    with pytest.raises(NotImplementedError, match="charged-cell external XC"):
        driver(ccm, name, max_iter=1)


@pytest.mark.parametrize("spin", (1, 2))
def test_external_real_gamma_requires_positive_becke_image_radius(spin):
    from vibeqc.periodic.ccm.direct import (
        run_ccm_rks_direct,
        run_ccm_uks_direct,
    )

    provider = _QuadraticFullGridProvider()
    name = _external_functional(provider)
    ccm = _compact_h2_ccm()
    driver = run_ccm_rks_direct if spin == 1 else run_ccm_uks_direct

    with pytest.raises(ValueError, match="becke_image_radius_bohr > 0"):
        driver(ccm, name, becke_image_radius_bohr=0.0, max_iter=1)


@pytest.mark.parametrize("functional", ["pbe", "pbe0"])
def test_uks_ccm_isolated_equals_molecular(functional):
    """Isolated Li (3e doublet): UKS-CCM aiccm2026dev-a == molecular run_uks."""
    cell = PeriodicSystem(3, np.diag([80.0, 80.0, 80.0]), [Atom(3, [0, 0, 0])], 0, 2)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    r = run_ccm_uks(ccm, functional, method="aiccm2026dev-a")
    mol = ccm.supercell
    opts = UKSOptions()
    opts.functional = functional
    ref = run_uks(mol, BasisSet(mol, "sto-3g"), opts)
    assert r.converged
    assert r.energy == pytest.approx(ref.energy, abs=1e-6)


def test_uks_ccm_closed_shell_matches_rks():
    """Closed-shell H2: UKS-CCM == RKS-CCM (aiccm2026dev-a, PBE)."""
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    u = run_ccm_uks(ccm, "pbe", method="aiccm2026dev-a").energy
    r = run_ccm_rks(ccm, "pbe", method="aiccm2026dev-a").energy
    assert u == pytest.approx(r, abs=1e-7)


def test_rks_ccm_unknown_method_raises():
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    with pytest.raises(ValueError, match="unknown CCM four-center method"):
        run_ccm_rks(ccm, "pbe", method="not_a_method")


def test_rks_ccm_unknown_four_center_raises():
    ccm = CCMSystem(_h2(), (1, 1, 1), "sto-3g")
    with pytest.raises(ValueError, match="unknown four_center"):
        run_ccm_rks(ccm, "pbe", four_center="sideways")


@pytest.mark.parametrize("method", ["union12", "aiccm2026dev-a"])
def test_rks_ccm_direct_matches_full(method):
    """KS-CCM Phase-3b: ``four_center="direct"`` (default, integral-direct J/K,
    O(nbf**2) memory) reproduces ``four_center="full"`` (the dense O(nbf**4)
    effective-tensor comparison reference) on a 1-D H₂ chain — same KS energy, for
    both four-center weights. This is what lets RKS/UKS-CCM scale to real 3-D
    cells; the ``"full"`` path stays runnable for comparison."""
    unit = PeriodicSystem(3, np.diag([6.0, 40.0, 40.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(unit, (3, 1, 1), "sto-3g")
    direct = run_ccm_rks(ccm, "pbe", method=method, four_center="direct")
    full = run_ccm_rks(ccm, "pbe", method=method, four_center="full")
    assert direct.converged and full.converged
    assert direct.energy == pytest.approx(full.energy, abs=1e-9)


@pytest.mark.slow
def test_rks_ccm_periodic_1d_converges():
    """1D H2 chain: RKS-CCM (aiccm2026dev-a, PBE) runs periodically and is bound."""
    unit = PeriodicSystem(3, np.diag([6.0, 40.0, 40.0]),
                          [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    ccm = CCMSystem(unit, (3, 1, 1), "sto-3g")
    r = run_ccm_rks(ccm, "pbe", method="aiccm2026dev-a")
    assert r.converged
    assert r.energy_per_atom < 0
