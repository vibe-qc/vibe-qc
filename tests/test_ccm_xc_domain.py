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
    assert not grouped.torus_complete       # the pair graph is the default
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


def _xc_builder(spin):
    from vibeqc.periodic.ccm import direct

    return (direct._real_gamma_rks_xc_builder if spin == 1
            else direct._real_gamma_uks_xc_builder)


def _lattice(cutoff_bohr):
    from vibeqc import LatticeSumOptions

    base = LatticeSumOptions()
    base.cutoff_bohr = float(cutoff_bohr)
    return base


@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("nrep", [(2, 1, 1), (2, 2, 1)])
def test_complete_torus_density_is_the_converged_domain(nrep, spin):
    """``torus_complete=True`` (#519 RG-1) sums every pair of images that
    reaches a grid batch through its torus residue (``periodic_xc.cpp``,
    ``xc_torus_complete_plan``) instead of the pair graph truncated at the XC
    image domain. At the #465 bound (14.0 bohr on this cube, so the default
    15-bohr list) that is the domain-converged XC term: E_xc and the folded
    V_xc equal the ungrouped kernel on a 30-bohr domain, where the pair graph
    at the bound itself sits about 3e-10 Ha away."""
    from vibeqc._vibeqc_core import Functional

    ccm = _h2_cube(nrep)
    func = Functional("pbe", spin)
    builder = _xc_builder(spin)
    complete = builder(ccm, func, lat_opts=_lattice(15.0),
                       rcut_strategy="pyscf_auto", torus_complete=True)
    converged = builder(ccm, func, lat_opts=_lattice(30.0), rcut_strategy="flat",
                        torus_grouping=False)
    truncated = builder(ccm, func, lat_opts=_lattice(15.0),
                        rcut_strategy="pyscf_auto", torus_grouping=False)
    assert complete.torus_complete and complete.xc_lattice_cutoff_bohr == 15.0
    D = _hcore_density(ccm)
    args = (D,) if spin == 1 else (0.5 * D, 0.5 * D)
    e1, *v1 = complete(*args)
    e0, *v0 = converged(*args)
    e2, *_ = truncated(*args)
    assert complete.torus_grouping_used
    assert abs(e1 - e0) < 1e-11, (e0, e1)
    for a, b in zip(v0, v1):
        assert np.max(np.abs(a - b)) < 1e-10
    assert abs(e2 - e0) > 1e-11  # the bound truncates; the control can fail


def test_complete_torus_density_refusals():
    """The complete torus density is refused where it is not defined: on a
    domain that does not close the density, without the torus-grouped kernel,
    and together with an XC source."""
    from vibeqc._vibeqc_core import Functional
    from vibeqc.periodic.ccm.direct import run_ccm_rks_direct

    ccm = _h2_cube((2, 1, 1))
    func = Functional("pbe", 1)
    builder = _xc_builder(1)
    with pytest.raises(ValueError, match="closes the density"):
        builder(ccm, func, lat_opts=_lattice(12.0), rcut_strategy="flat",
                torus_complete=True)
    with pytest.raises(ValueError, match="torus-grouped kernel"):
        builder(ccm, func, lat_opts=_lattice(15.0), torus_grouping=False,
                torus_complete=True)
    with pytest.raises(ValueError, match="xc_source"):
        run_ccm_rks_direct(ccm, "pbe", xc_torus_complete=True,
                           xc_source=object())


@pytest.mark.slow
@pytest.mark.parametrize("spin", [1, 2])
def test_torus_grouping_matches_at_the_grown_domain(spin):
    """The same identities at the #465 image bound on LiH/STO-3G, the cell whose
    diffuse Li 2sp shells defeat the AO-reach screen (the domain the grouping
    exists to make affordable). The default pair-graph cover equals the
    ungrouped kernel to rounding. The opt-in complete torus density adds the
    pairs beyond the closure: E_xc moves by 4.9e-11 Ha on the HCORE density and
    V_xc by 2.0e-9, the bound's own truncation; on the converged state the SCF
    energy moves by 2.9e-11 Ha/cell. Its XC build is 18 times faster than the
    pair-graph cover there (4.2 s against 75.9 s on six threads)."""
    from vibeqc import LatticeSumOptions
    from vibeqc._vibeqc_core import Functional
    from vibeqc.periodic.ccm import direct

    ccm = _lih((1, 1, 1))
    func = Functional("pbe", spin)
    base = LatticeSumOptions()
    base.cutoff_bohr = 15.0
    builder = (direct._real_gamma_rks_xc_builder if spin == 1
               else direct._real_gamma_uks_xc_builder)
    kw = dict(lat_opts=base, rcut_strategy="pyscf_auto")
    D = _hcore_density(ccm)
    args = (D,) if spin == 1 else (0.5 * D, 0.5 * D)
    e0, *v0 = builder(ccm, func, torus_grouping=False, **kw)(*args)
    complete = builder(ccm, func, torus_complete=True, **kw)
    pair_graph = builder(ccm, func, **kw)
    assert complete.torus_complete and not pair_graph.torus_complete
    e1, *v1 = complete(*args)
    e2, *v2 = pair_graph(*args)
    assert complete.torus_grouping_used and pair_graph.torus_grouping_used
    assert abs(e2 - e0) < 1e-10, (e0, e2)
    for a, b in zip(v0, v2):
        assert np.max(np.abs(a - b)) < 1e-10
    assert abs(e1 - e0) < 1e-10, (e0, e1)
    for a, b in zip(v0, v1):
        assert np.max(np.abs(a - b)) < 1e-8
