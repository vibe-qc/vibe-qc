"""Raw native 3D domain/state contracts; run on a containing QC runtime.

These operator checks do not replace the NaH four-guess convergence tests.
"""
import numpy as np
import pytest
import vibeqc as vq


def _system_basis():
    core = vq._vibeqc_core
    system = vq.PeriodicSystem(3, np.eye(3) * 8., [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.5]),
    ])
    basis = core.BasisSet(system.unit_cell_molecule(), [
        core.ShellInfo(0, 0, True, [.7], [1.], [0., 0., 0.]),
        core.ShellInfo(1, 0, True, [.1], [1.], [0., 0., 3.5]),
    ], "native-domain-contract", coefficients_pre_normalized=True)
    return system, basis


def _fold(operator, points):
    blocks = np.asarray(operator.blocks)
    origins = np.asarray([cell.r_cart for cell in operator.cells])
    return np.einsum("kg,gij->kij", np.exp(1j * np.asarray(points) @ origins.T), blocks)


@pytest.mark.parametrize("method", ["krhf", "krks"])
def test_native_3d_trace_uses_actual_density_and_raw_fock(method):
    """One evaluated READ state: no convergence/DIIS can conceal its residual."""
    core = vq._vibeqc_core
    system, basis = _system_basis()
    mesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.PeriodicSCFOptions() if method == "krhf" else vq.PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = opts.lattice_opts.nuclear_cutoff_bohr = 10.
    physical = vq.LatticeSumOptions()
    physical.cutoff_bohr = physical.nuclear_cutoff_bohr = 10.
    physical.pair_complete_1e = True
    overlap = _fold(core.compute_overlap_lattice(basis, system, physical), mesh.kpoints)[0]
    d = np.array([[.8, .2], [.2, .5]], dtype=complex)
    d *= 2. / np.trace(d @ overlap).real
    opts.initial_guess = vq.InitialGuess.READ
    opts.read_density_k = [d]
    opts.max_iter = 1
    opts.use_diis = False
    opts.damping = opts.fock_mixing = opts.level_shift = 0.
    if method == "krks":
        opts.functional = "lda"
    driver = core.run_rhf_periodic if method == "krhf" else core.run_rks_periodic
    result = driver(system, basis, mesh, opts)
    assert not opts.lattice_opts.pair_complete_1e  # local resolution only
    np.testing.assert_allclose(result.overlap, overlap, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.density, d.real, atol=1e-12, rtol=0)
    f = np.asarray(result.fock)
    np.testing.assert_allclose(f, f.T, atol=1e-10, rtol=0)
    residual = np.linalg.norm(f @ d @ overlap - overlap @ d @ f)
    assert residual > 1e-5  # discriminates the old cell-count factor
    assert result.scf_trace[0].grad_norm == pytest.approx(residual, abs=1e-10)


def test_native_physical_jk_complex_k_adjoint_and_density_derivative():
    core = vq._vibeqc_core
    system, basis = _system_basis()
    mesh = vq.monkhorst_pack(system, [1, 1, 3])
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = opts.nuclear_cutoff_bohr = 10.
    opts.pair_complete_1e = True
    cells = core.physical_eri_lattice_cells(basis, system, 10., 10.)
    points = np.asarray(mesh.kpoints)
    weights = np.asarray(mesh.weights)
    d = np.tile(np.array([[.8, .2], [.2, .5]], complex), (len(points), 1, 1))
    a = np.tile(np.array([[.3, -.1], [-.1, -.4]], complex), (len(points), 1, 1))
    for state, scale in ((d, .11), (a, .07)):
        state[:, 0, 1] += 1j * scale * np.sin(8. * points[:, 2])
        state[:, 1, 0] = state[:, 0, 1].conj()

    def jk(state):
        density = core.real_space_density_from_density_matrices(list(state), mesh, cells)
        result = core.build_jk_2e_real_space(basis, system, opts, density)
        return _fold(result.J, points), _fold(result.K, points)

    def dot(left, right):
        return np.einsum("k,kij,kij->", weights, left.conj(), right).real

    jd, kd = jk(d)
    ja, ka = jk(a)
    for matrix in (jd, kd, ja, ka):
        np.testing.assert_allclose(matrix, matrix.swapaxes(1, 2).conj(), atol=1e-10, rtol=0)
    assert abs(dot(d, ja) - dot(a, jd)) < 1e-10
    assert abs(dot(d, ka) - dot(a, kd)) < 1e-10
    h = _fold(core.compute_kinetic_lattice(basis, system, opts), points)
    h += _fold(core.compute_nuclear_lattice(basis, system, opts), points)
    step = 1e-4
    energies = []
    for state in (d + step * a, d - step * a):
        j, k = jk(state)
        energies.append(dot(state, h) + .5 * dot(state, j - .5 * k))
    assert abs((energies[0] - energies[1]) / (2 * step) - dot(a, h + jd - .5 * kd)) < 1e-8


@pytest.mark.parametrize("guess", ["HCORE", "SAD", "SAP", "MINAO", "HUECKEL", "PATOM"])
def test_native_krks_guesses_share_complete_xc_domain(guess):
    """Every seed reaches one physical XC evaluation on the common domain."""
    core = vq._vibeqc_core
    system, basis = _system_basis()
    mesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = opts.lattice_opts.nuclear_cutoff_bohr = 10.
    opts.functional = "lda"
    opts.initial_guess = getattr(vq.InitialGuess, guess)
    opts.max_iter = 1
    opts.use_diis = False
    opts.grid.n_radial, opts.grid.n_theta, opts.grid.n_phi = 12, 4, 8
    result = core.run_rks_periodic(system, basis, mesh, opts)
    assert result.n_iter == 1
    assert np.isfinite(result.energy)
    assert np.isfinite(result.scf_trace[0].grad_norm)
    np.testing.assert_allclose(result.fock, np.asarray(result.fock).T, atol=1e-10, rtol=0)


def test_native_krks_fixed_state_energy_derivative_uses_complete_xc_potential():
    """Extra difference blocks must carry the variational XC adjoint too."""
    core = vq._vibeqc_core
    system, basis = _system_basis()
    mesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = opts.lattice_opts.nuclear_cutoff_bohr = 10.
    opts.functional = "lda"
    opts.initial_guess = vq.InitialGuess.READ
    opts.max_iter = 1
    opts.use_diis = False
    opts.damping = opts.fock_mixing = opts.level_shift = 0.
    opts.grid.n_radial, opts.grid.n_theta, opts.grid.n_phi = 12, 4, 8
    physical = vq.LatticeSumOptions()
    physical.cutoff_bohr = physical.nuclear_cutoff_bohr = 10.
    physical.pair_complete_1e = True
    overlap = _fold(core.compute_overlap_lattice(basis, system, physical), mesh.kpoints)[0].real
    d = np.array([[.8, .2], [.2, .5]])
    d *= 2. / np.trace(d @ overlap)
    direction = np.array([[.3, -.1], [-.1, -.4]])
    direction -= .5 * np.trace(direction @ overlap) * d
    assert abs(np.trace(direction @ overlap)) < 1e-12

    def evaluate(state):
        opts.read_density_k = [state.astype(complex)]
        return core.run_rks_periodic(system, basis, mesh, opts)

    center = evaluate(d)
    step = 1e-4
    difference = (evaluate(d + step * direction).energy
                  - evaluate(d - step * direction).energy) / (2 * step)
    analytic = np.einsum("ij,ij->", direction, center.fock)
    assert difference == pytest.approx(analytic, abs=1e-7)
