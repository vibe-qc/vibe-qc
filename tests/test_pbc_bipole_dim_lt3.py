"""dim<3 (1D/2D) coverage for the four ``run_pbc_bipole_*`` drivers.

For dim<3 the BIPOLE drivers fall back to a direct-truncated (non-Ewald)
gauge — the documented "direct-only path for dim<3 diagnostic runs"
(``pbc_bipole.py`` docstring). Two bugs used to crash that path
(fixed 2026-05-31, found in the 2026-05-30/31 end-to-end audit):

  * ``compute_ext_el_spheropole`` — a 3D-Ewald reciprocal-space (K=0
    limit) correction that raises for dim!=3 — was called
    unconditionally from every driver;
  * RKS/UHF/UKS left ``e_j_multipole`` unbound in the direct branch, so
    the Fock build raised ``UnboundLocalError`` before the driver even
    reached the spheropole call (RHF already initialised it).

These tests pin the fixed diagnostic behaviour: all four low-level drivers
run on 1D and 2D systems, the spheropole term is absent (``None``) in the
direct gauge, and the isolated-cell checks are padding-independent and tend
to the molecular RHF result. They do **not** establish a physical compact-cell
1-D/2-D Coulomb model or cutoff-converged periodic energies. Public periodic
jobs and ``PeriodicSCFProvider`` therefore refuse this route at dim<3 (IID
542); the low-level entry points remain available for diagnostics only.

Note: ``use_ewald_j_split=True`` is *explicitly* rejected for dim<3 (see
``test_use_ewald_j_split_dim2_raises`` in
``test_pbc_bipole_ewald_split_integration.py``); the auto path used here
selects ``use_ewald_j_split=False`` for dim<3.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import (
    PeriodicKSOptions,
    PeriodicRHFOptions,
    RHFOptions,
)
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks


def _h2_chain(dim, a=8.0, vac=25.0):
    """Closed-shell H2 unit (re=1.4 bohr) periodic in ``dim`` directions.

    The non-periodic directions get a ``vac``-bohr box; the diagnostic
    direct-truncated sum is invariant to that padding (see the corresponding
    test).
    """
    if dim == 1:
        lat = np.diag([a, vac, vac])
    elif dim == 2:
        lat = np.diag([a, a, vac])
    else:  # pragma: no cover - guard
        raise ValueError(f"dim must be 1 or 2; got {dim}")
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])]
    system = vq.PeriodicSystem(dim, lat, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _hf_opts(cutoff=8.0, max_iter=3):
    o = PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = cutoff
    o.max_iter = max_iter
    o.use_diis = False
    o.initial_guess = InitialGuess.SAD
    return o


def _ks_opts(cutoff=8.0, max_iter=3):
    o = PeriodicKSOptions()
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = cutoff
    o.max_iter = max_iter
    o.use_diis = False
    o.initial_guess = InitialGuess.SAD
    return o


def _run(driver, system, basis, *, max_iter=3):
    kmesh = monkhorst_pack(system, [1, 1, 1])
    if driver in (run_pbc_bipole_rhf, run_pbc_bipole_uhf):
        return driver(system, basis, kmesh, _hf_opts(max_iter=max_iter), progress=False)
    return driver(
        system,
        basis,
        kmesh,
        _ks_opts(max_iter=max_iter),
        functional="pbe",
        progress=False,
    )


ALL_DRIVERS = [
    pytest.param(run_pbc_bipole_rhf, id="rhf"),
    pytest.param(run_pbc_bipole_rks, id="rks"),
    pytest.param(run_pbc_bipole_uhf, id="uhf"),
    pytest.param(run_pbc_bipole_uks, id="uks"),
]


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("driver", ALL_DRIVERS)
def test_bipole_driver_runs_dim_lt3_without_crash(driver, dim):
    """All four drivers complete on a 1D/2D H2 system; the spheropole
    term is absent (None) in the direct gauge.

    Regression for the unconditional ``compute_ext_el_spheropole`` call
    (ValueError) and the ``e_j_multipole`` UnboundLocalError in
    RKS/UHF/UKS — both used to crash this path for every driver.
    """
    system, basis = _h2_chain(dim)
    result = _run(driver, system, basis)

    assert result.n_iter >= 1
    assert np.isfinite(result.energy)
    # Spheropole is a 3D-Ewald-only correction — absent for dim<3.
    assert result.e_ext_el_spheropole is None
    comp = result.energy_components[-1]
    assert comp.e_ext_el_spheropole is None
    # Energy-component bookkeeping closes with the spheropole as 0.
    assert math.isclose(
        comp.e_electronic
        + comp.e_nuclear_repulsion
        + (comp.e_ext_el_spheropole or 0.0),
        comp.e_total,
        abs_tol=1e-8,
    )
    assert math.isclose(comp.e_total, result.energy, abs_tol=1e-8)


def _converged_hf_opts(cutoff):
    o = _hf_opts(cutoff=cutoff, max_iter=100)
    o.use_diis = True
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-7
    return o


def test_bipole_rhf_dim1_energy_is_vacuum_independent():
    """A correct direct-truncated 1D sum runs only along the periodic
    axis, so the energy must not depend on the non-periodic box size.

    This pins dimensional bookkeeping, not compact-cell cutoff convergence.
    """
    energies = []
    for vac in (20.0, 35.0):
        system, basis = _h2_chain(1, a=8.0, vac=vac)
        kmesh = monkhorst_pack(system, [1, 1, 1])
        result = run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            _converged_hf_opts(cutoff=12.0),
            use_ewald_j_split=False,
            progress=False,
        )
        assert result.converged
        energies.append(result.energy)
    assert abs(energies[0] - energies[1]) < 1e-8


def test_bipole_rhf_dim1_isolated_limit_matches_molecular_rhf():
    """A far-spaced 1D chain (a=30 bohr, cells non-interacting) gives a
    per-cell energy equal to the molecular RHF energy of the unit cell.

    Absolute-energy cross-check for the isolated-cell diagnostic against
    molecular C++ RHF; this does not certify compact-cell cutoff convergence.
    """
    system, basis = _h2_chain(1, a=30.0, vac=30.0)
    kmesh = monkhorst_pack(system, [1, 1, 1])
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _converged_hf_opts(cutoff=14.0),
        use_ewald_j_split=False,
        progress=False,
    )
    assert result.converged

    mol_result = vq.run_rhf(system.unit_cell_molecule(), basis, RHFOptions())
    assert abs(result.energy - mol_result.energy) < 1e-7


def test_bipole_rhf_uhf_agree_on_closed_shell_dim1():
    """UHF must reduce to RHF on a closed-shell 1D system (sanity that
    both spin paths take the fixed direct gauge consistently).

    Regression for the 2026-06-09 merge artifact that dropped the
    ``_zero_cross_cell_density`` Γ-locality projection from the RHF
    driver: without it the Γ-only Fock build overcounts exchange across
    cell pairs (RHF landed 0.26 Ha *below* UHF on this closed-shell
    system, which is variationally impossible for a correct solver)."""
    system, basis = _h2_chain(1)
    r_rhf = _run(run_pbc_bipole_rhf, system, basis, max_iter=5)
    r_uhf = _run(run_pbc_bipole_uhf, system, basis, max_iter=5)
    assert math.isclose(r_rhf.energy, r_uhf.energy, abs_tol=1e-8)


@pytest.mark.parametrize("dim", [1, 2])
def test_bipole_pbe0_rks_uks_closed_shell_and_component_closure(dim):
    """The direct-gauge hybrid must use the same spin-resolved K algebra."""
    system, basis = _h2_chain(dim)
    kmesh = monkhorst_pack(system, [1, 1, 1])

    rks_opts = _ks_opts(max_iter=3)
    rks_opts.functional = "pbe0"
    uks_opts = _ks_opts(max_iter=3)
    uks_opts.functional = "pbe0"
    rks = run_pbc_bipole_rks(
        system,
        basis,
        kmesh,
        rks_opts,
        functional="pbe0",
        progress=False,
    )
    uks = run_pbc_bipole_uks(
        system,
        basis,
        kmesh,
        uks_opts,
        functional="pbe0",
        progress=False,
    )

    for result in (rks, uks):
        comp = result.energy_components[-1]
        e_j = (
            (comp.e_j_short_range or 0.0)
            + (comp.e_j_long_range or 0.0)
            + (comp.e_j_multipole or 0.0)
        )
        assert comp.e_exchange is not None
        assert math.isclose(result.e_coulomb, e_j, abs_tol=1e-10)
        assert math.isclose(
            result.e_hf_exchange,
            comp.e_exchange,
            abs_tol=1e-10,
        )
        assert math.isclose(
            result.e_coulomb + result.e_hf_exchange,
            comp.e_two_electron,
            abs_tol=1e-10,
        )

    assert math.isclose(rks.energy, uks.energy, abs_tol=1e-8)
    assert math.isclose(rks.e_electronic, uks.e_electronic, abs_tol=1e-8)
    assert math.isclose(rks.e_xc, uks.e_xc, abs_tol=1e-8)
    assert math.isclose(rks.e_coulomb, uks.e_coulomb, abs_tol=1e-8)
    assert math.isclose(
        rks.e_hf_exchange,
        uks.e_hf_exchange,
        abs_tol=1e-8,
    )
    np.testing.assert_allclose(uks.fock_alpha[0], rks.fock[0], atol=1e-8)
    np.testing.assert_allclose(uks.fock_beta[0], rks.fock[0], atol=1e-8)
