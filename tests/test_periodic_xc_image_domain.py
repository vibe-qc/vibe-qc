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
