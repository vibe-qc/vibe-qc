"""Streamed fixed-density RI Coulomb nuclear derivatives, not full SCF forces."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, _vibeqc_core as core


_POSITIONS = np.array([[.1, -.2, .3], [.7, .4, -.8], [-.4, 1.1, .6]])


def _system(positions=_POSITIONS, pure=True, separate=False, s_only=False):
    molecule = vq.Molecule([vq.Atom(2, p.tolist()) for p in np.asarray(positions)], 0, 1)
    bases = []
    for auxiliary in (False, True):
        shells = []
        owners = (1, 2) if auxiliary and separate else (0,) if separate else (0, 1, 2)
        for i in owners:
            s = core.ShellInfo()
            s.atom_index, s.origin, s.pure = i, np.asarray(positions)[i].tolist(), pure
            s.l = 0 if s_only else ((0, 2, 1) if auxiliary else (0, 1, 0))[i]
            s.exponents = [.55+.17*i, 1.3+.25*i]
            s.coefficients = [.7, -.19] if auxiliary else [.8, .11]
            shells.append(s)
        bases.append(vq.BasisSet(molecule, shells, "manta-ri-gradient-fixture", False))
    return molecule, *bases


def _density(n):
    a = np.random.default_rng(109).normal(size=(n, n))
    return (a+a.T)/2


def _request(kernel):
    r = manta.KernelRequest()
    r.kernel, r.omega = kernel, 0. if kernel == "coulomb" else .65
    return r


def _options(orbital, auxiliary, request, truncate=False):
    o = manta.RIOptions()
    if truncate:
        spectrum = np.linalg.eigvalsh(manta.RIJKBuilder(orbital, auxiliary, request=request).metric)
        gaps = np.flatnonzero(np.diff(spectrum) > 1e-5*spectrum[-1])
        k = min(gaps, key=lambda i: abs(i-(len(spectrum)/2-1)))
        o.absolute_cutoff = (spectrum[k]+spectrum[k+1])/2
    return o


def _numpy_energy(orbital, auxiliary, d, options):
    # Independent libint raw tensors plus NumPy spectral inverse. Small oracle
    # fixtures only; the native consumer never allocates the three-index tensor.
    v = vq.compute_2c_eri(auxiliary)
    b = np.einsum("pij,ij->p", vq.compute_3c_eri(orbital, auxiliary), d)
    lam, u = np.linalg.eigh(v)
    keep = lam > max(options.absolute_cutoff, options.relative_cutoff*max(0., lam[-1]))
    return .5*np.sum((u[:, keep].T @ b)**2 / lam[keep])


@pytest.mark.parametrize("kernel", ["coulomb", "erf", "erfc"])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("truncate", [False, True])
def test_gradient_against_displaced_ri_energies_and_actual_work(kernel, pure, truncate):
    mol, orb, aux = _system(pure=pure)
    request = _request(kernel)
    options = _options(orb, aux, request, truncate)
    d = _density(orb.nbasis)
    plan = manta.plan_ri_coulomb_gradient(mol, orb, aux, options, request)
    result = manta.compute_ri_coulomb_gradient(mol, orb, aux, d, options, request)
    builder = manta.RIJKBuilder(orb, aux, options, request)
    energy = builder.build(d).receipt.coulomb_energy
    assert result.receipt.coulomb_energy == pytest.approx(energy, abs=2e-12)
    assert result.receipt.ri_source_identity == builder.source_identity
    assert result.metric_response.factor_identity == builder.metric_info.factor_identity
    assert result.metric_response.rank == builder.metric_info.rank
    if truncate:
        assert 0 < result.metric_response.rank < aux.nbasis
        assert result.receipt.metric_residual_norm > 1e-5
    np.testing.assert_allclose(result.values, result.coupling+result.metric, atol=1e-14)
    np.testing.assert_allclose(result.values.sum(axis=0), 0., atol=3e-11)
    h = 1e-4
    fd = np.zeros((3, 3))
    for atom in range(3):
        for axis in range(3):
            energies = []
            for k in (-2, -1, 1, 2):
                positions = _POSITIONS.copy(); positions[atom, axis] += k*h
                _, displaced_orb, displaced_aux = _system(positions, pure)
                if kernel == "coulomb":
                    energies.append(_numpy_energy(displaced_orb, displaced_aux, d, options))
                else:
                    ri = manta.RIJKBuilder(displaced_orb, displaced_aux, options, request)
                    assert ri.metric_info.rank == result.metric_response.rank
                    energies.append(ri.build(d).receipt.coulomb_energy)
            fd[atom, axis] = (energies[0]-8*energies[1]+8*energies[2]-energies[3])/(12*h)
    np.testing.assert_allclose(result.values, fd, atol=8e-8, rtol=2e-7)
    r = result.receipt
    assert r.metric_shell_calls == aux.nshells**2
    assert r.coupling_shell_calls == aux.nshells*orb.nshells**2
    assert r.metric_derivative_shell_calls == 6*aux.nshells**2
    assert r.coupling_derivative_shell_calls == 9*aux.nshells*orb.nshells**2
    assert (r.metric_shell_calls+r.coupling_shell_calls+r.metric_derivative_shell_calls+r.coupling_derivative_shell_calls) == plan.shell_calls
    assert r.primitive_integrals_evaluated <= plan.primitive_evaluations_upper_bound
    assert r.primitive_integrals_evaluated >= 4*r.metric_shell_calls+8*r.coupling_shell_calls+4*r.metric_derivative_shell_calls+8*r.coupling_derivative_shell_calls
    assert r.integral_elements_consumed == 7*aux.nbasis**2+10*aux.nbasis*orb.nbasis**2
    assert r.modeled_peak_bytes == plan.modeled_peak_bytes
    assert r.metric_residual_evaluated


@pytest.mark.parametrize("separate", [False, True])
def test_metric_and_coupling_components_separately_against_libint(separate):
    mol, orb, aux = _system(separate=separate)
    d, req = _density(orb.nbasis), _request("coulomb")
    o = _options(orb, aux, req, truncate=True)
    result = manta.compute_ri_coulomb_gradient(mol, orb, aux, d, o)
    def source(orbital, auxiliary):
        v = vq.compute_2c_eri(auxiliary)
        lam, u = np.linalg.eigh(v)
        keep = lam > max(o.absolute_cutoff, o.relative_cutoff*max(0., lam[-1]))
        g = (u[:, keep]/lam[keep]) @ u[:, keep].T
        b = np.einsum("pij,ij->p", vq.compute_3c_eri(orbital, auxiliary), d)
        return g, b
    g0, b0 = source(orb, aux)
    h = 1e-4
    metric_fd, coupling_fd = np.zeros((3, 3)), np.zeros((3, 3))
    for atom in range(3):
        for axis in range(3):
            em, eb = [], []
            for k in (-2, -1, 1, 2):
                pos = _POSITIONS.copy(); pos[atom, axis] += k*h
                _, xorb, xaux = _system(pos, separate=separate)
                g, b = source(xorb, xaux)
                em.append(.5*b0 @ g @ b0); eb.append(.5*b @ g0 @ b)
            metric_fd[atom, axis] = (em[0]-8*em[1]+8*em[2]-em[3])/(12*h)
            coupling_fd[atom, axis] = (eb[0]-8*eb[1]+8*eb[2]-eb[3])/(12*h)
    np.testing.assert_allclose(result.metric, metric_fd, atol=5e-8, rtol=2e-7)
    np.testing.assert_allclose(result.coupling, coupling_fd, atol=5e-8, rtol=2e-7)
    if separate:
        np.testing.assert_array_equal(result.metric[0], np.zeros(3))
        assert np.linalg.norm(result.coupling[0]) > 1e-6


def test_s_distribution_rotation_and_translation_sum_rules():
    mol, orb, aux = _system(s_only=True)
    d = _density(orb.nbasis)
    r = manta.compute_ri_coulomb_gradient(mol, orb, aux, d)
    np.testing.assert_allclose(np.cross(_POSITIONS, r.values).sum(axis=0), 0., atol=1e-11)
    q, _ = np.linalg.qr(np.random.default_rng(834).normal(size=(3, 3)))
    rmol, rorb, raux = _system(_POSITIONS @ q.T + np.array([.8, -.2, 1.1]), s_only=True)
    rotated = manta.compute_ri_coulomb_gradient(rmol, rorb, raux, d)
    np.testing.assert_allclose(rotated.values, r.values @ q.T, atol=1e-11)


@pytest.mark.parametrize("factor", [0., -.7, 2.3])
def test_density_scaling_ownership_and_source_evidence(factor):
    mol, orb, aux = _system()
    d = _density(orb.nbasis)
    r = manta.compute_ri_coulomb_gradient(mol, orb, aux, d)
    scaled = manta.compute_ri_coulomb_gradient(mol, orb, aux, np.asfortranarray(factor*d))
    np.testing.assert_allclose(scaled.values, factor**2*r.values, atol=5e-12)
    assert scaled.receipt.coulomb_energy == pytest.approx(factor**2*r.receipt.coulomb_energy, abs=5e-12)
    assert scaled.receipt.source_identity == r.receipt.source_identity
    assert scaled.receipt.density_identity != r.receipt.density_identity
    before = r.values.copy(); d[:] = np.nan
    np.testing.assert_array_equal(r.values, before)
    assert len(r.metric_response.source_identity) == 64


def test_resolved_zero_rank_returns_zero_without_derivative_passes():
    mol, orb, aux = _system()
    o = manta.RIOptions(); o.absolute_cutoff = 1e6
    r = manta.compute_ri_coulomb_gradient(mol, orb, aux, _density(orb.nbasis), o)
    assert r.metric_response.rank == 0
    np.testing.assert_array_equal(r.values, np.zeros((3, 3)))
    assert r.receipt.coulomb_energy == 0.
    assert r.receipt.coupling_shell_calls == r.receipt.metric_derivative_shell_calls == r.receipt.coupling_derivative_shell_calls == 0
    assert not r.receipt.metric_residual_evaluated
    assert r.receipt.metric_shell_calls == aux.nshells**2


def test_rank_crossing_refuses_gradient_without_silent_fixed_projector():
    mol, orb, aux = _system()
    o = manta.RIOptions()
    o.absolute_cutoff = np.linalg.eigvalsh(manta.RIJKBuilder(orb, aux).metric)[4]
    with pytest.raises(RuntimeError, match="rank cutoff"):
        manta.compute_ri_coulomb_gradient(mol, orb, aux, _density(orb.nbasis), o)


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_evaluations_upper_bound"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_aggregate_resource_caps_before_density_copy(field, plan_field):
    mol, orb, aux = _system()
    b = manta.ResourceBudget(); b.other_live_bytes = 32123
    p = manta.plan_ri_coulomb_gradient(mol, orb, aux, budget=b)
    setattr(b, field, getattr(p, plan_field))
    r = manta.compute_ri_coulomb_gradient(mol, orb, aux, _density(orb.nbasis), budget=b)
    assert r.receipt.modeled_peak_bytes == p.modeled_peak_bytes
    setattr(b, field, getattr(p, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.compute_ri_coulomb_gradient(mol, orb, aux, np.full((orb.nbasis, orb.nbasis), np.nan), budget=b)


@pytest.mark.parametrize("kind", ["float32", "complex", "strided", "sequence", "nan", "asymmetric", "dimension"])
def test_density_contract(kind):
    mol, orb, aux = _system(); n = orb.nbasis; d = np.eye(n)
    bad = {"float32": d.astype(np.float32), "complex": d.astype(complex),
           "strided": np.eye(2*n)[::2, ::2], "sequence": d.tolist(), "nan": d*np.nan,
           "asymmetric": np.triu(np.ones((n, n))), "dimension": np.eye(n+1)}[kind]
    with pytest.raises((ValueError, TypeError)):
        manta.compute_ri_coulomb_gradient(mol, orb, aux, bad)


@pytest.mark.parametrize("which", ["orbital", "auxiliary", "molecule"])
def test_geometry_source_mismatch(which):
    mol, orb, aux = _system()
    shifted = _POSITIONS.copy(); shifted[1, 0] += .01
    xmol, xorb, xaux = _system(shifted)
    if which == "orbital":
        orb = xorb
    elif which == "auxiliary":
        aux = xaux
    else:
        mol = xmol
    with pytest.raises(ValueError, match="centers/atom map"):
        manta.compute_ri_coulomb_gradient(mol, orb, aux, _density(orb.nbasis))


@pytest.mark.parametrize("field,value", [("kernel", "yukawa"), ("scalar", "complex"), ("derivative", 1), ("centers", 3)])
def test_unsupported_requests_refuse(field, value):
    mol, orb, aux = _system(); request = _request("coulomb"); setattr(request, field, value)
    with pytest.raises(ValueError):
        manta.compute_ri_coulomb_gradient(mol, orb, aux, _density(orb.nbasis), request=request)


def test_ri_coulomb_gradient_citation_route():
    from vibeqc.output.citations.registry import load_default_database
    refs = load_default_database().assemble(method="RHF", extra_methods=["manta_ri_coulomb_gradient"])
    keys = {entry.key for entry in refs.citations}
    assert {"higham_matrix_functions_2008", "neese_shark_2023", "mcmurchie_davidson_1978"} <= keys


def _exchange_numpy(orbital, auxiliary, alpha, beta, options, *, metric=None, coupling=None):
    v = vq.compute_2c_eri(auxiliary) if metric is None else metric
    b = vq.compute_3c_eri(orbital, auxiliary) if coupling is None else coupling
    lam, u = np.linalg.eigh(v)
    keep = lam > max(options.absolute_cutoff, options.relative_cutoff*max(0., lam[-1]))
    g = (u[:, keep]/lam[keep]) @ u[:, keep].T
    energy = 0.
    for d in (alpha, beta):
        k = np.einsum("pq,pik,kl,qjl->ij", g, b, d, b)
        energy -= .5*np.trace(d @ k)
    return energy


@pytest.mark.parametrize("kernel", ["coulomb", "erf", "erfc"])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("truncate", [False, True])
def test_exchange_gradient_finite_differences_spin_sources_and_counts(kernel, pure, truncate):
    mol, orb, aux = _system(pure=pure)
    request = _request(kernel)
    o = _options(orb, aux, request, truncate); o.mode_block = 2
    alpha, beta = _density(orb.nbasis), .3*np.eye(orb.nbasis)
    plan = manta.plan_ri_exchange_gradient(mol, orb, aux, o, request)
    result = manta.compute_ri_exchange_gradient(mol, orb, aux, alpha, beta, o, request)
    ri = manta.RIJKBuilder(orb, aux, o, request)
    pair = ri.build_exchange_pair(alpha, beta)
    expected_energy = -.5*(np.trace(alpha @ pair.K_alpha)+np.trace(beta @ pair.K_beta))
    r = result.receipt
    assert r.exchange_energy == pytest.approx(expected_energy, abs=2e-11)
    assert r.ri_source_identity == ri.source_identity
    assert r.density_identity == pair.receipt.density_identity
    assert tuple(r.density_identities) == tuple(pair.density_identities)
    assert result.metric_response.factor_identity == ri.metric_info.factor_identity
    np.testing.assert_allclose(result.values, result.coupling+result.metric, atol=1e-14)
    np.testing.assert_allclose(result.values.sum(axis=0), 0., atol=4e-11)
    h = 1e-4
    fd = np.zeros((3, 3))
    for atom in range(3):
        for axis in range(3):
            energies = []
            for step in (-2, -1, 1, 2):
                pos = _POSITIONS.copy(); pos[atom, axis] += step*h
                _, ob, ab = _system(pos, pure)
                if kernel == "coulomb":
                    energies.append(_exchange_numpy(ob, ab, alpha, beta, o))
                else:
                    builder = manta.RIJKBuilder(ob, ab, o, request)
                    assert builder.metric_info.rank == result.metric_response.rank
                    k = builder.build_exchange_pair(alpha, beta)
                    energies.append(-.5*(np.trace(alpha @ k.K_alpha)+np.trace(beta @ k.K_beta)))
            fd[atom, axis] = (energies[0]-8*energies[1]+8*energies[2]-energies[3])/(12*h)
    np.testing.assert_allclose(result.values, fd, atol=8e-8, rtol=2e-7)
    blocks = (aux.nbasis+o.mode_block-1)//o.mode_block
    assert plan.auxiliary_blocks == blocks
    assert r.coupling_passes == plan.maximum_coupling_passes == blocks*(blocks+1)//2
    assert r.coupling_derivative_passes == plan.maximum_derivative_passes == blocks
    assert r.metric_shell_calls == aux.nshells**2
    assert r.coupling_shell_calls == r.coupling_passes*aux.nshells*orb.nshells**2
    assert r.metric_derivative_shell_calls == 6*aux.nshells**2
    assert r.coupling_derivative_shell_calls == blocks*9*aux.nshells*orb.nshells**2
    assert r.metric_shell_calls+r.coupling_shell_calls+r.metric_derivative_shell_calls+r.coupling_derivative_shell_calls == plan.shell_calls
    assert r.primitive_integrals_evaluated <= plan.primitive_evaluations_upper_bound
    assert r.integral_elements_consumed == 7*aux.nbasis**2+(r.coupling_passes+9*blocks)*aux.nbasis*orb.nbasis**2
    assert r.modeled_peak_bytes == plan.modeled_peak_bytes


@pytest.mark.parametrize("block", [1, 3, 8, 64])
def test_exchange_block_size_changes_work_not_energy_gradient_or_source(block):
    mol, orb, aux = _system()
    alpha, beta = _density(orb.nbasis), np.diag(np.linspace(.1, .7, orb.nbasis))
    small = manta.RIOptions(); small.mode_block = 2
    large = manta.RIOptions(); large.mode_block = block
    a = manta.compute_ri_exchange_gradient(mol, orb, aux, alpha, beta, small)
    b = manta.compute_ri_exchange_gradient(mol, orb, aux, alpha, beta, large)
    np.testing.assert_allclose(a.values, b.values, atol=5e-12)
    assert a.receipt.exchange_energy == pytest.approx(b.receipt.exchange_energy, abs=5e-12)
    assert a.receipt.source_identity == b.receipt.source_identity
    assert a.receipt.ri_source_identity == b.receipt.ri_source_identity
    assert a.receipt.modeled_peak_bytes != b.receipt.modeled_peak_bytes


@pytest.mark.parametrize("separate", [False, True])
def test_exchange_metric_and_coupling_components_with_truncated_subspace(separate):
    mol, orb, aux = _system(separate=separate)
    alpha, beta = _density(orb.nbasis), .4*np.eye(orb.nbasis)
    o = _options(orb, aux, _request("coulomb"), True)
    result = manta.compute_ri_exchange_gradient(mol, orb, aux, alpha, beta, o)
    v0, b0 = vq.compute_2c_eri(aux), vq.compute_3c_eri(orb, aux)
    h = 1e-4
    fm, fb = np.zeros((3, 3)), np.zeros((3, 3))
    for atom in range(3):
        for axis in range(3):
            em, eb = [], []
            for k in (-2, -1, 1, 2):
                pos = _POSITIONS.copy(); pos[atom, axis] += k*h
                _, ob, ab = _system(pos, separate=separate)
                em.append(_exchange_numpy(ob, ab, alpha, beta, o, coupling=b0))
                eb.append(_exchange_numpy(ob, ab, alpha, beta, o, metric=v0))
            fm[atom, axis] = (em[0]-8*em[1]+8*em[2]-em[3])/(12*h)
            fb[atom, axis] = (eb[0]-8*eb[1]+8*eb[2]-eb[3])/(12*h)
    np.testing.assert_allclose(result.metric, fm, atol=5e-8, rtol=2e-7)
    np.testing.assert_allclose(result.coupling, fb, atol=5e-8, rtol=2e-7)
    if separate:
        np.testing.assert_array_equal(result.metric[0], np.zeros(3))


def test_exchange_spin_additivity_quadratic_scaling_zero_rank_and_rotation():
    mol, orb, aux = _system(s_only=True)
    alpha, beta, zero = _density(orb.nbasis), np.eye(orb.nbasis), np.zeros((orb.nbasis, orb.nbasis))
    total = manta.compute_ri_exchange_gradient(mol, orb, aux, alpha, beta)
    a = manta.compute_ri_exchange_gradient(mol, orb, aux, alpha, zero)
    b = manta.compute_ri_exchange_gradient(mol, orb, aux, zero, np.asfortranarray(beta))
    swapped = manta.compute_ri_exchange_gradient(mol, orb, aux, beta, alpha)
    scaled = manta.compute_ri_exchange_gradient(mol, orb, aux, -.3*alpha, -.3*beta)
    np.testing.assert_allclose(total.values, a.values+b.values, atol=2e-12)
    np.testing.assert_allclose(total.values, swapped.values, atol=2e-12)
    np.testing.assert_allclose(.09*total.values, scaled.values, atol=2e-12)
    np.testing.assert_allclose(np.cross(_POSITIONS, total.values).sum(axis=0), 0., atol=2e-12)
    q, _ = np.linalg.qr(np.random.default_rng(91).normal(size=(3, 3)))
    rmol, rorb, raux = _system(_POSITIONS @ q.T, s_only=True)
    rotated = manta.compute_ri_exchange_gradient(rmol, rorb, raux, alpha, beta)
    np.testing.assert_allclose(rotated.values, total.values @ q.T, atol=3e-12)
    o = manta.RIOptions(); o.absolute_cutoff = 1e6
    empty = manta.compute_ri_exchange_gradient(mol, orb, aux, alpha, beta, o)
    np.testing.assert_array_equal(empty.values, np.zeros((3, 3)))
    assert empty.receipt.exchange_energy == empty.receipt.coupling_passes == empty.receipt.coupling_derivative_passes == 0
    assert empty.receipt.metric_shell_calls == aux.nshells**2
    zeros = manta.compute_ri_exchange_gradient(mol, orb, aux, zero, zero)
    np.testing.assert_array_equal(zeros.values, np.zeros((3, 3)))


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_evaluations_upper_bound"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_exchange_aggregate_resource_caps_before_two_density_copies(field, plan_field):
    mol, orb, aux = _system(); o = manta.RIOptions(); o.mode_block = 2
    b = manta.ResourceBudget(); b.other_live_bytes = 12345
    p = manta.plan_ri_exchange_gradient(mol, orb, aux, o, budget=b)
    setattr(b, field, getattr(p, plan_field))
    d = _density(orb.nbasis)
    result = manta.compute_ri_exchange_gradient(mol, orb, aux, d, d, o, budget=b)
    assert result.receipt.modeled_peak_bytes == p.modeled_peak_bytes
    setattr(b, field, getattr(p, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.compute_ri_exchange_gradient(mol, orb, aux, d*np.nan, d*np.nan, o, budget=b)


@pytest.mark.parametrize("spin", [0, 1])
@pytest.mark.parametrize("bad", ["float32", "complex", "strided", "nan", "asymmetric", "dimension", "sequence"])
def test_exchange_strict_spin_density_contracts(spin, bad):
    mol, orb, aux = _system(); n = orb.nbasis; good = np.eye(n)
    value = {"float32": good.astype(np.float32), "complex": good.astype(complex),
             "strided": np.eye(2*n)[::2, ::2], "nan": good*np.nan,
             "asymmetric": np.triu(np.ones((n, n))), "dimension": np.eye(n+1), "sequence": good.tolist()}[bad]
    densities = [good, good]; densities[spin] = value
    with pytest.raises((ValueError, TypeError)):
        manta.compute_ri_exchange_gradient(mol, orb, aux, *densities)


def test_exchange_rank_crossing_and_geometry_mismatch_refuse():
    mol, orb, aux = _system(); d = _density(orb.nbasis)
    o = manta.RIOptions(); o.absolute_cutoff = np.linalg.eigvalsh(manta.RIJKBuilder(orb, aux).metric)[4]
    with pytest.raises(RuntimeError, match="rank cutoff"):
        manta.compute_ri_exchange_gradient(mol, orb, aux, d, d, o)
    shifted = _POSITIONS.copy(); shifted[0, 0] += .1
    _, _, bad_aux = _system(shifted)
    with pytest.raises(ValueError, match="centers/atom map"):
        manta.compute_ri_exchange_gradient(mol, orb, bad_aux, d, d)


def test_ri_exchange_gradient_citation_route():
    from vibeqc.output.citations.registry import load_default_database
    refs = load_default_database().assemble(method="RHF", extra_methods=["manta_ri_exchange_gradient"])
    assert {"higham_matrix_functions_2008", "neese_shark_2023", "mcmurchie_davidson_1978"} <= {entry.key for entry in refs.citations}
