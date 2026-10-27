"""Focused finite-pair Hartree contract and public opt-in acceptance."""
import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_gapw_pair_domain import PairDomainJBuilder, resolve_options
from vibeqc.periodic_gapw_grid import make_grid
from vibeqc.periodic_gapw_j import evaluate_gpw_energy, run_periodic_rhf_gpw


def fixture():
    system = vq.PeriodicSystem(3, np.eye(3)*6.,
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    opts = vq.LatticeSumOptions()
    opts.pair_complete_1e = True
    opts.cutoff_bohr = 10.
    return system, basis, opts, make_grid(np.eye(3)*6., cutoff_ha=10.)


def test_pair_hartree_adjoint_and_energy_derivative():
    system, basis, opts, grid = fixture()
    j = PairDomainJBuilder(basis, system, grid, opts)
    d = np.array([[.8, .2], [.2, .4]])
    h = np.array([[.1, -.3], [-.3, .2]])
    jd, jh = j.build_J(d), j.build_J(h)
    assert np.sum(d*jh) == pytest.approx(np.sum(h*jd), abs=1e-12)
    step = 1e-4
    energy = lambda x: .5*np.sum(x*j.build_J(x))
    assert (energy(d+step*h)-energy(d-step*h))/(2*step) == pytest.approx(np.sum(h*jd), abs=1e-10)
    with pytest.raises(ValueError): j.build_J(np.array([[1., .2], [0., 1.]]))
    with pytest.raises(ValueError): j.build_J(d.astype(complex))


def test_option_snapshot_and_unmatched_consumers_refused():
    _, _, opts, _ = fixture()
    copied = resolve_options(opts)
    opts.cutoff_bohr = 11.
    assert copied.cutoff_bohr == 10.
    for kwargs in ({'functional':'tpss'}, {'omega':.2}, {'cache':object()}):
        with pytest.raises(NotImplementedError): resolve_options(opts, **kwargs)
    opts.cutoff_bohr = -1.
    with pytest.raises(ValueError): resolve_options(opts)


def test_public_scf_and_energy_use_same_pair_operator():
    system, basis, opts, grid = fixture()
    result = run_periodic_rhf_gpw(system, basis, grid=grid, lattice_options=opts,
        v_ne_convention='smeared_erfc', smearing_alpha=1., initial_guess='HCORE',
        max_iter=50, quiet=True)
    assert result.converged
    density = result.density_matrix if hasattr(result, 'density_matrix') else result.density
    evaluated = evaluate_gpw_energy(system, basis, density, grid=grid,
        lattice_options=opts, v_ne_convention='smeared_erfc', smearing_alpha=1., quiet=True)
    assert result.energy == pytest.approx(evaluated.e_total, abs=1e-9)
    assert opts.pair_complete_1e and opts.cutoff_bohr == 10.


def test_pair_fft_adjoint_including_even_grid_nyquist(monkeypatch):
    from types import SimpleNamespace
    from vibeqc.periodic_gapw_pair_domain import core
    grid = SimpleNamespace(shape=(4,4,4), n_points=64,
        voxel_volume_bohr3=.5, lattice_bohr=np.eye(3))
    frequencies = np.fft.fftfreq(4)
    squared = sum(g*g for g in np.meshgrid(frequencies, frequencies, frequencies, indexing='ij'))
    kernel = np.divide(4*np.pi, squared, out=np.zeros_like(squared), where=squared!=0)
    monkeypatch.setattr(core, 'solve_poisson_coulomb',
        lambda rho, lattice: np.fft.ifftn(np.fft.fftn(rho)*kernel).real)
    builder = PairDomainJBuilder.__new__(PairDomainJBuilder)
    builder.basis, builder.grid = SimpleNamespace(nbasis=2), grid
    random = np.random.default_rng(413)
    pair = random.normal(size=(2,2,64)) + 1j*random.normal(size=(2,2,64))
    pair = .5*(pair+pair.transpose(1,0,2))
    builder._panels = lambda: iter([(0,64,pair)])
    d, h = np.array([[.8,.2],[.2,.4]]), np.array([[.1,-.3],[-.3,.2]])
    jd, jh = builder.build_J(d), builder.build_J(h)
    assert np.sum(d*jh) == pytest.approx(np.sum(h*jd), abs=1e-12)
    energy = lambda x: .5*np.sum(x*builder.build_J(x))
    assert (energy(d+1e-4*h)-energy(d-1e-4*h))/2e-4 == pytest.approx(np.sum(h*jd), abs=1e-9)


@pytest.mark.parametrize('functional', ['lda', 'pbe'])
def test_scalar_pair_xc_density_derivative(functional):
    system, basis, options, grid = fixture()
    builder = PairDomainJBuilder(basis, system, grid, options)
    density, direction = np.eye(2), np.array([[.1, .03], [.03, -.1]])
    func = vq.Functional(functional, 1)
    energy, potential = builder.scalar_xc(density, func)
    step = 1e-5
    plus = builder.scalar_xc(density+step*direction, func)[0]
    minus = builder.scalar_xc(density-step*direction, func)[0]
    assert (plus-minus)/(2*step) == pytest.approx(np.sum(potential*direction), abs=1e-8)


def test_corrected_exchange_is_self_adjoint():
    system, basis, options, grid = fixture()
    builder = PairDomainJBuilder(basis, system, grid, options, exchange=True)
    density, direction = np.eye(2), np.array([[.1, .03], [.03, -.1]])
    kd, kh = builder.build_K(density), builder.build_K(direction)
    assert np.sum(density*kh) == pytest.approx(np.sum(direction*kd), abs=1e-10)
    assert builder.source_identity()['exchange']['kernel'] == 'corrected-ewald'
    energy = lambda d: -.25*np.sum(d*builder.build_K(d))
    step = 1e-4
    assert (energy(density+step*direction)-energy(density-step*direction))/(2*step) == pytest.approx(-.5*np.sum(direction*kd), abs=1e-9)
