"""Source controls for primitive-cell XC support delivered by both CCM closures.

Native basis, grid and XC evaluation are doubles; these are not energy-parity
or native-kernel acceptance tests. The domain bound and torus maps are real.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.experimental


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("nrep", [(2, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("strategy,cutoff", [("pyscf_auto", 15.0), ("flat", 15.0), ("pyscf_auto", 40.0)])
def test_real_gamma_delivers_primitive_xc_domain(monkeypatch, spin, nrep, strategy, cutoff):
    from vibeqc import _vibeqc_core as core
    from vibeqc import lattice_screening, periodic_grid, periodic_k_gdf, periodic_xc_domain
    from vibeqc.periodic.ccm import direct

    class Options:
        cutoff_bohr = 15.0
        becke_image_radius_bohr = 0.0
        pair_complete_1e = False

    molecule = object()
    unit = SimpleNamespace(dim=3, unit_cell_molecule=lambda: molecule)
    shell = SimpleNamespace(exponents=[0.05], coefficients=[1.0], l=0, origin=[0., 0., 0.])
    basis = SimpleNamespace(nbasis=1, shells=lambda: [shell])
    ccm = SimpleNamespace(unit_system=unit, basis_name="synthetic", nrep=nrep,
                          n_cells=int(np.prod(nrep)))
    seen = []

    def basis_for(mol, name):
        assert mol is molecule  # Never evaluate a supercell-span bound.
        return basis

    def cells_for(system, radius):
        assert system is unit
        indices = [(0, 0, 0), (1, 0, 0)]
        if radius > 15.0:
            indices += [(-3, 0, 0), (4, 1, 0)]
        return [SimpleNamespace(index=index) for index in indices]

    def matrix_set(n_mu, cells, blocks):
        assert n_mu == 1
        return SimpleNamespace(cells=cells, blocks=blocks)

    def make_opts(basis_arg, *, strategy, base_opts, cutoff_bohr):
        assert basis_arg is basis
        out = Options()
        out.__dict__.update(vars(base_opts))
        out.cutoff_bohr = cutoff_bohr
        return out

    def xc(*args):
        ubasis, system, grid, func = args[:4]
        densities, options, domain = args[4:-2], args[-2], args[-1]
        assert ubasis is basis and system is unit and grid == "primitive-grid"
        assert domain == core.PeriodicXCDensityDomain.AUTO
        seen.append((densities, options))
        return SimpleNamespace(e_xc=2.0, V_xc=densities[0],
                               V_alpha=densities[0], V_beta=densities[-1])

    monkeypatch.setattr(core, "LatticeSumOptions", Options)
    monkeypatch.setattr(core, "BasisSet", basis_for)
    monkeypatch.setattr(core, "direct_lattice_cells", cells_for)
    monkeypatch.setattr(core, "make_lattice_matrix_set", matrix_set)
    monkeypatch.setattr(core, "build_xc_periodic", xc)
    monkeypatch.setattr(core, "build_xc_periodic_uks", xc)
    monkeypatch.setattr(lattice_screening, "make_lattice_opts", make_opts)
    monkeypatch.setattr(periodic_k_gdf, "_direct_cells", cells_for)
    monkeypatch.setattr(periodic_grid, "build_periodic_becke_grid",
                        lambda *a, **kw: "primitive-grid")
    base = Options()
    base.cutoff_bohr = cutoff
    builder = (direct._real_gamma_rks_xc_builder if spin == 1
               else direct._real_gamma_uks_xc_builder)
    # Domain delivery only: the synthetic system cannot carry the kernel's
    # finite-torus regrouping (a real mesh and lattice); that path is checked
    # against the ungrouped kernel on real cells below.
    closure = builder(ccm, SimpleNamespace(is_external=False), lat_opts=base,
                      rcut_strategy=strategy, torus_grouping=False)
    density = np.arange(ccm.n_cells**2, dtype=float).reshape(ccm.n_cells, ccm.n_cells)
    density = 0.5 * (density + density.T)
    result = closure(density) if spin == 1 else closure(density / 3., 2. * density / 3.)
    bloch_opts, bloch_cells = periodic_k_gdf._xc_image_domain(
        unit, basis, base, cells_for(unit, cutoff), is_ks=True, periodic_grid=True,
        rcut_strategy=strategy, density_domain=core.PeriodicXCDensityDomain.AUTO)
    bound = periodic_xc_domain.xc_image_cutoff_bound(unit, basis)
    assert bound > 15.0
    expected = max(cutoff, bound) if strategy == "pyscf_auto" else cutoff
    densities, delivered = seen[0]
    assert delivered.cutoff_bohr == expected == bloch_opts.cutoff_bohr
    assert delivered.becke_image_radius_bohr == expected
    assert closure.xc_lattice_cutoff_bohr == expected
    assert [c.index for c in densities[0].cells] == [c.index for c in bloch_cells]
    assert len(densities) == spin
    assert result[0] == 2.0 * ccm.n_cells
    assert all(v.shape == density.shape and np.array_equal(v, v.T) for v in result[1:])
    assert base.cutoff_bohr == cutoff and base.becke_image_radius_bohr == 0.0


# ---------------------------------------------------------------------------
# The finite-torus regrouping of the XC kernel (#519, RG-1)
# ---------------------------------------------------------------------------

def _h2_cube(nrep):
    from vibeqc import Atom, PeriodicSystem
    from vibeqc.periodic.ccm import CCMSystem

    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _lih(nrep):
    from vibeqc import Atom, PeriodicSystem
    from vibeqc.periodic.ccm import CCMSystem

    a = 4.0840 / 0.529177210903
    lat = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    cell = PeriodicSystem(
        3, lat, [Atom(3, [0.0, 0.0, 0.0]),
                 Atom(1, [0.5 * a, 0.5 * a, 0.5 * a])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _hcore_density(ccm):
    from vibeqc.periodic.ccm.direct import ccm_direct_oneelectron

    S, h, _ = ccm_direct_oneelectron(ccm)
    w, U = np.linalg.eigh(S)
    X = U / np.sqrt(w)
    _, Cp = np.linalg.eigh(X.T @ h @ X)
    C = X @ Cp
    n_occ = ccm.supercell.n_electrons() // 2
    return 2.0 * C[:, :n_occ] @ C[:, :n_occ].T


def _grouped_and_reference(ccm, spin, *, rcut_strategy, cutoff_bohr):
    from vibeqc import LatticeSumOptions
    from vibeqc._vibeqc_core import Functional
    from vibeqc.periodic.ccm import direct

    func = Functional("pbe", spin)
    base = LatticeSumOptions()
    base.cutoff_bohr = float(cutoff_bohr)
    builder = (direct._real_gamma_rks_xc_builder if spin == 1
               else direct._real_gamma_uks_xc_builder)
    kw = dict(lat_opts=base, rcut_strategy=rcut_strategy)
    grouped = builder(ccm, func, **kw)
    reference = builder(ccm, func, torus_grouping=False, **kw)
    assert grouped.torus_grouping and not reference.torus_grouping
    D = _hcore_density(ccm)
    args = (D,) if spin == 1 else (0.5 * D, 0.5 * D)
    e1, *v1 = grouped(*args)
    e0, *v0 = reference(*args)
    assert grouped.torus_grouping_used and not reference.torus_grouping_used
    return e0, e1, v0, v1


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("system,nrep", [("h2", (2, 1, 1)), ("h2", (2, 2, 2)),
                                         ("lih", (1, 1, 1))])
def test_torus_grouping_matches_the_ungrouped_kernel(system, nrep, spin):
    """``build_xc_periodic_kpoints`` regroups the retained bra-ket image pairs
    by torus residue (``periodic_xc.cpp``, ``xc_torus_plan``: a disjoint cover
    of the same pair graph under the same screen), so on the real-Gamma
    projected density E_xc and the folded V_xc equal the ungrouped real-space
    kernel to rounding. The default closure groups; ``torus_grouping=False``
    is the reference switch."""
    ccm = _h2_cube(nrep) if system == "h2" else _lih(nrep)
    e0, e1, v0, v1 = _grouped_and_reference(
        ccm, spin, rcut_strategy="flat", cutoff_bohr=15.0)
    assert abs(e1 - e0) < 1e-10, (e0, e1)
    for a, b in zip(v0, v1):
        assert np.max(np.abs(a - b)) < 1e-10


@pytest.mark.slow
@pytest.mark.parametrize("spin", [1, 2])
def test_torus_grouping_matches_at_the_grown_domain(spin):
    """The same identity at the #465 image bound on LiH/STO-3G, the cell whose
    diffuse Li 2sp shells defeat the AO-reach screen (the domain the grouping
    exists to make affordable)."""
    e0, e1, v0, v1 = _grouped_and_reference(
        _lih((1, 1, 1)), spin, rcut_strategy="pyscf_auto", cutoff_bohr=15.0)
    assert abs(e1 - e0) < 1e-10, (e0, e1)
    for a, b in zip(v0, v1):
        assert np.max(np.abs(a - b)) < 1e-10
