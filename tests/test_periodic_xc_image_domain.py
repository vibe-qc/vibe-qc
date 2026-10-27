"""XC density image domain of the 3-D GDF Kohn-Sham routes (#465).

The GDF KS drivers handed ``build_xc_periodic`` the caller's lattice cutoff
(15 bohr by default) while their one-electron domain grows to a converged
bound, so the periodic XC density was truncated on compact cells: the
cubic Al2O3 model (STO-3G, RKS-PBE at Gamma) sat 1.55 mHa above PySCF and
above its own 20-bohr value. ``vibeqc.periodic_xc_domain`` now grows the XC
cutoff to the reach of AO pairs that overlap at one grid point (amplitude
1e-4), under the default ``rcut_strategy`` only.

The gates here are the bound itself, the convergence of E_xc at that bound
on a fixed density (cheap: coarse grid, no SCF), and the wiring of the
grown domain into both GDF KS drivers.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem, make_basis
from vibeqc._vibeqc_core import (
    Functional,
    GridOptions,
    LatticeSumOptions,
    PeriodicKSOptions,
    build_xc_periodic,
    compute_overlap_lattice,
)
from vibeqc.periodic_grid import build_periodic_becke_grid
from vibeqc.periodic_rhf_gdf import _density_set_torus_gamma
from vibeqc.periodic_xc_domain import (
    xc_image_cutoff_bound,
    xc_lattice_opts,
    xc_lattice_opts_at,
)

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def _al2o3_cubic():
    """The cubic Al2O3 stuffed-fluorite model of the MgO/Al2O3 parity script."""
    a = 5.05 * BOHR_PER_ANGSTROM
    al = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
    o = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    atoms = [Atom(13, [x * a, y * a, z * a]) for x, y, z in al]
    atoms += [Atom(8, [x * a, y * a, z * a]) for x, y, z in o]
    return PeriodicSystem(3, np.diag([a, a, a]), atoms)


def _base_opts(cutoff: float = 15.0) -> LatticeSumOptions:
    opts = LatticeSumOptions()
    opts.cutoff_bohr = cutoff
    opts.nuclear_cutoff_bohr = 25.0
    return opts


def test_bound_and_strategy_gate():
    system = _al2o3_cubic()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    bound = xc_image_cutoff_bound(system, basis)
    # Al 3s/3p reach 7.5 bohr at amplitude 1e-4; span is the Al-Al diagonal.
    assert 22.5 < bound < 24.0

    base = _base_opts()
    grown = xc_lattice_opts(system, basis, base, rcut_strategy="pyscf_auto")
    assert grown is not base
    assert grown.cutoff_bohr == pytest.approx(bound)
    assert grown.nuclear_cutoff_bohr == base.nuclear_cutoff_bohr
    assert not grown.pair_complete_1e
    assert base.cutoff_bohr == 15.0

    # An explicit flat strategy keeps the caller's finite domain.
    assert xc_lattice_opts(system, basis, base, rcut_strategy="flat") is base
    assert xc_lattice_opts(system, basis, base, rcut_strategy=None) is base
    # A caller cutoff already past the bound is kept as given.
    wide = _base_opts(40.0)
    assert xc_lattice_opts(system, basis, wide, rcut_strategy="pyscf_auto") is wide


def test_recorded_cutoff_rebuild():
    system = _al2o3_cubic()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    base = _base_opts()
    # Results from before #465 and HF results record no grown domain.
    assert xc_lattice_opts_at(basis, base, None) is base
    assert xc_lattice_opts_at(basis, base, 0.0) is base
    assert xc_lattice_opts_at(basis, base, 15.0) is base
    rebuilt = xc_lattice_opts_at(basis, base, 23.0)
    assert rebuilt.cutoff_bohr == 23.0
    assert not rebuilt.pair_complete_1e


def test_non_3d_cell_keeps_caller_domain():
    a = 4.0
    system = PeriodicSystem(
        2, np.diag([a, a, 20.0]), [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])]
    )
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    base = _base_opts()
    assert xc_lattice_opts(system, basis, base, rcut_strategy="pyscf_auto") is base


def test_fixed_density_exc_converges_at_bound():
    """E_xc of one fixed Gamma-torus density on the Al2O3 cell: the 15-bohr
    domain is mHa-scale off, the grown domain matches a wider one."""
    system = _al2o3_cubic()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    grid_options = GridOptions()
    grid_options.n_radial = 30
    grid_options.n_theta = 8
    grid_options.n_phi = 16
    grid = build_periodic_becke_grid(
        system,
        grid_options=grid_options,
        image_radius_bohr=float(PeriodicKSOptions().becke_image_radius_bohr),
    )
    func = Functional("pbe", 1)
    # A diagonal AO density keeps rho(r) = sum_mu n_mu (sum_R chi_mu)^2
    # non-negative everywhere; 50 electrons over 33 AOs.
    D = np.diag(np.full(basis.nbasis, 50.0 / basis.nbasis))

    def e_xc(opts):
        D_set = _density_set_torus_gamma(compute_overlap_lattice(basis, system, opts), D)
        return float(build_xc_periodic(basis, system, grid, func, D_set, opts).e_xc)

    grown = xc_lattice_opts(system, basis, _base_opts(), rcut_strategy="pyscf_auto")
    e_default = e_xc(_base_opts())
    e_grown = e_xc(grown)
    e_wide = e_xc(xc_lattice_opts_at(basis, _base_opts(), grown.cutoff_bohr + 6.0))
    assert abs(e_default - e_wide) > 1e-4
    assert e_grown == pytest.approx(e_wide, abs=1e-8)


class _Stop(Exception):
    pass


def _spy(monkeypatch, module, name, seen):
    def spy(*args, **kwargs):
        opts = kwargs.get("lat_opts", args[-1] if args else None)
        seen.append(float(opts.cutoff_bohr))
        raise _Stop

    monkeypatch.setattr(module, name, spy)


@pytest.mark.parametrize("route", ["gamma", "kmesh"])
def test_ks_drivers_hand_grown_domain_to_xc(monkeypatch, route):
    """Both GDF KS drivers evaluate XC on the grown domain (the first XC
    call is intercepted, so no SCF runs). H2/STO-3G in an 8-bohr cube at an
    8-bohr cutoff keeps the fit cheap while the bound (14.0 bohr) exceeds
    the cutoff."""
    import vibeqc._vibeqc_core as core
    import vibeqc.periodic_k_gdf as k_gdf
    from vibeqc.pbc_gdf import run_pbc_gdf_rks

    system = PeriodicSystem(
        3, np.eye(3) * 8.0, [Atom(1, [4.0, 4.0, 3.3]), Atom(1, [4.0, 4.0, 4.7])]
    )
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.lattice_opts.nuclear_cutoff_bohr = 16.0
    bound = xc_image_cutoff_bound(system, basis)
    assert bound > 8.0
    seen: list[float] = []
    with pytest.raises(_Stop):
        if route == "gamma":
            _spy(monkeypatch, core, "build_xc_periodic", seen)
            run_pbc_gdf_rks(system, basis, opts, functional="lda", progress=False)
        else:
            _spy(monkeypatch, k_gdf, "_build_xc_k_from_density", seen)
            k_gdf.run_krks_periodic_gdf(
                system, basis, (2, 1, 1), opts, functional="lda",
                use_compcell=False, progress=False,
            )
    assert seen == [pytest.approx(bound)]


@pytest.mark.parametrize("requested", [0.0, -1.0, 4.0, 30.0])
@pytest.mark.parametrize("physical", [False, True])
def test_bipole_xc_support_preserves_integral_and_explicit_domains(requested, physical):
    from vibeqc.periodic_xc_domain import bipole_xc_lattice_opts

    system = PeriodicSystem(3, np.eye(3)*8., [Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])])
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    base = _base_opts(8.)
    base.pair_complete_1e = physical
    base.becke_image_radius_bohr = min(requested, 8.) if requested > 0 else 8.
    result = bipole_xc_lattice_opts(system, basis, base, requested_radius=requested)
    if physical and requested <= 0:
        assert result is not base
        assert result.cutoff_bohr == xc_image_cutoff_bound(system, basis)
        assert result.becke_image_radius_bohr == result.cutoff_bohr
        assert result.pair_complete_1e
        assert result.nuclear_cutoff_bohr == base.nuclear_cutoff_bohr
    else:
        assert result is base
    assert base.cutoff_bohr == 8.


@pytest.mark.parametrize("driver", ["rks", "uks"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
@pytest.mark.parametrize("functional", ["pbe", "pbe0", "tpss", "m06-l", "r2scan"])
def test_bipole_xc_expanded_domain_matches_wider_density_reference(
    driver, mesh, functional, record_property,
):
    import importlib
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_external_xc import periodic_xc_difference_closed_cells
    from vibeqc.periodic_k_density import real_space_density_from_per_k_density
    from vibeqc.pbc_bipole_common import bvk_torus_density_matrices

    system = PeriodicSystem(3, np.eye(3)*8., [Atom(1, [4., 4., 3.3]), Atom(1, [4., 4., 4.7])])
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    options = PeriodicKSOptions()
    options.functional = functional
    options.initial_guess = core.InitialGuess.HCORE
    options.lattice_opts.cutoff_bohr = 8.
    options.lattice_opts.nuclear_cutoff_bohr = 16.
    options.grid.n_radial = 20
    options.grid.n_theta = 8
    options.grid.n_phi = 16
    options.max_iter = 80
    options.conv_tol_energy = 1e-10
    options.conv_tol_grad = 1e-8
    kmesh = core.monkhorst_pack(system, list(mesh))
    run = getattr(importlib.import_module('vibeqc.pbc_bipole_'+driver), 'run_pbc_bipole_'+driver)
    result = run(system, basis, kmesh, options, progress=False, ewald_precision=1e-10)
    assert result.converged
    domain = result.resolved_domain
    assert dict(domain.lattice_1e.values)['cutoff_bohr'] == 8.
    assert dict(domain.lattice_2e.values)['cutoff_bohr'] == 8.
    xc_opts = domain.xc_lattice.restore(LatticeSumOptions)
    assert xc_opts.cutoff_bohr == xc_image_cutoff_bound(system, basis)
    assert options.lattice_opts.becke_image_radius_bohr == 0.
    densities = [result.density] if driver == 'rks' else [result.density_alpha, result.density_beta]
    per_k = [bvk_torus_density_matrices(d, result.kpoints_cart, mesh) for d in densities]
    grid = build_periodic_becke_grid(system, grid_options=options.grid,
                                   image_radius_bohr=options.becke_image_radius_bohr)
    func = Functional(functional, 1 if driver == 'rks' else 2)

    def evaluate(radius):
        opts = domain.xc_lattice.restore(LatticeSumOptions)
        opts.cutoff_bohr = opts.becke_image_radius_bohr = radius
        cells = periodic_xc_difference_closed_cells(system, radius, radius)
        folded = [real_space_density_from_per_k_density(d, kmesh, cells) for d in per_k]
        build = core.build_xc_periodic if driver == 'rks' else core.build_xc_periodic_uks
        return float(build(basis, system, grid, func, *folded, opts,
                           core.PeriodicXCDensityDomain.PERIODIC_LATTICE).e_xc)

    old = evaluate(8.)
    at_bound = evaluate(xc_opts.cutoff_bohr)
    wide = evaluate(xc_opts.cutoff_bohr + 6.)
    assert result.e_xc == pytest.approx(at_bound, rel=0., abs=1e-10)
    assert at_bound == pytest.approx(wide, rel=0., abs=1e-7)
    assert abs(old - wide) > 1e-6
    record_property('energy_Ha', result.energy)
    record_property('xc_radius_bohr', xc_opts.cutoff_bohr)
    record_property('old_fixed_density_error_Ha', old-wide)
    record_property('new_fixed_density_error_Ha', at_bound-wide)


    # Independent periodic AO evaluator and libxc on the identical quadrature
    # and accepted D(k): isolates XC support from Hartree and grid errors.
    pyscf = pytest.importorskip("pyscf")
    from pyscf.pbc import gto, dft
    from pyscf.dft import libxc
    cell = gto.Cell()
    cell.a = np.eye(3)*8.
    cell.atom = [('H', (4.,4.,3.3)), ('H', (4.,4.,4.7))]
    cell.unit = 'Bohr'
    cell.basis = 'sto-3g'
    cell.precision = 1e-12
    cell.verbose = 0
    cell.build()
    coords = np.asarray(grid.points)
    is_mgga = libxc.xc_type(functional) == "MGGA"
    spin_rho = [np.zeros((6 if is_mgga else 4, len(coords))) for _ in per_k]
    for ik, (weight, kpoint) in enumerate(zip(kmesh.weights, kmesh.kpoints)):
        ao = dft.numint.eval_ao(cell, coords, kpt=np.asarray(kpoint), deriv=1)
        for rho, blocks in zip(spin_rho, per_k):
            d = np.asarray(blocks[ik])
            rho[0] += weight*np.einsum('pi,ij,pj->p', ao[0].conj(), d, ao[0]).real
            for axis in range(3):
                rho[axis+1] += 2*weight*np.einsum(
                    'pi,ij,pj->p', ao[axis+1].conj(), d, ao[0]).real
                if is_mgga:
                    rho[5] += .5*weight*np.einsum(
                        'pi,ij,pj->p', ao[axis+1].conj(), d, ao[axis+1]).real
    rho_total = sum(rho[0] for rho in spin_rho)
    exc = libxc.eval_xc(functional, spin_rho[0] if driver=='rks' else np.asarray(spin_rho),
                        spin=0 if driver=='rks' else 1, deriv=0)[0]
    reference = float(np.dot(np.asarray(grid.weights), rho_total*exc))
    assert at_bound == pytest.approx(reference, rel=0., abs=1e-7)
    record_property('pyscf_same_grid_xc_Ha', reference)
    record_property('pyscf_same_grid_xc_error_Ha', at_bound-reference)
