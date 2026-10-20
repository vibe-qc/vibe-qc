"""Stationary high-spin ROHF on the retained, finite classic chi Hamiltonian.

The orbital optimization is chi-owned. Only numerical AO support, J/K and the
shared molecular Roothaan effective-Fock primitive are reused. No CCM or UHF
SCF construction supplies this reference. Guest and Saunders, Mol. Phys. 28,
819 (1974), doi:10.1080/00268977400102171, Eqs. (8), (12), (13), (15),
define the within-category freedom and the three independent orbital gradients.
"""
from types import SimpleNamespace
from time import perf_counter
import numbers

import numpy as np

ROHF_CONVENTION = 'coulson-roothaan-average-diagonal/v1'
ROHF_CANONICALIZATION = 'closed-open-virtual-average-fock/v1'


def _hermitian(a):
    return .5 * (a + a.conj().swapaxes(-1, -2))


def _subspaces(nmo, n_alpha, n_beta):
    return dict(closed=tuple(range(n_beta)), open=tuple(range(n_beta, n_alpha)),
                virtual=tuple(range(n_alpha, nmo)))


def _counts(n_alpha, n_beta, nmo):
    if (any(isinstance(n, bool) or not isinstance(n, numbers.Integral)
            for n in (n_alpha, n_beta)) or not 0 <= n_beta <= n_alpha <= nmo):
        raise ValueError('ROHF requires integer 0 <= n_beta <= n_alpha <= retained rank')


def rohf_densities(coefficients, n_alpha, n_beta):
    c = np.asarray(coefficients)
    _counts(n_alpha, n_beta, c.shape[-1])
    return tuple(c[:, :, :n] @ c[:, :, :n].conj().transpose(0, 2, 1)
                 for n in (n_alpha, n_beta))


def rohf_gradient(coefficients, fock_alpha, fock_beta, n_alpha, n_beta):
    """Return the anti-Hermitian physical orbital gradient (unweighted).

    For C -> C exp(K), dE = Re tr(G^H K). G_pq =
    sum_sigma (n_q_sigma-n_p_sigma) F_pq_sigma. Independent lower
    blocks are Fb_oc, (Fa+Fb)_vc, Fa_vo. The full Frobenius product
    counts both conjugate blocks; independent real rotation derivatives
    therefore contain an additional factor two. These are the complex
    extension of Guest--Saunders (1974) Eq. (12); stationarity is Eq. (13),
    or Knowles et al. (1991), doi:10.1016/S0009-2614(91)85118-G, Eqs. (6-8).
    Neither individual spin Fock must commute with its occupied projector.
    """
    c = np.asarray(coefficients)
    _counts(n_alpha, n_beta, c.shape[-1])
    grad = np.zeros((len(c), c.shape[-1], c.shape[-1]), dtype=complex)
    for f, count in ((fock_alpha, n_alpha), (fock_beta, n_beta)):
        occ = (np.arange(c.shape[-1]) < count).astype(float)
        fm = c.conj().transpose(0, 2, 1) @ np.asarray(f) @ c
        grad += fm * (occ[None, None, :] - occ[None, :, None])
    return grad


def rohf_residuals(coefficients, fock_alpha, fock_beta, n_alpha, n_beta, weights):
    c = np.asarray(coefficients)
    fa = c.conj().transpose(0, 2, 1) @ np.asarray(fock_alpha) @ c
    fb = c.conj().transpose(0, 2, 1) @ np.asarray(fock_beta) @ c
    blocks = dict(closed_open_residual=fb[:, :n_beta, n_beta:n_alpha],
                  closed_virtual_residual=(fa+fb)[:, :n_beta, n_alpha:],
                  open_virtual_residual=fa[:, n_beta:n_alpha, n_alpha:])
    out = {key: float(np.max(abs(value), initial=0.)) for key, value in blocks.items()}
    g = rohf_gradient(c, fock_alpha, fock_beta, n_alpha, n_beta)
    out['generalized_rohf_gradient_norm'] = float(np.sqrt(
        np.einsum('k,kpq,kpq->', np.asarray(weights), g.conj(), g).real))
    return out


def canonicalize_rohf(coefficients, fock_alpha, fock_beta, n_alpha, n_beta):
    """Choose Fc eigenvectors within C/O/V only; never optimize occupations.

    Guest--Saunders Eq. (8) permits arbitrary Hermitian operators within each
    category. Here Z=Fc for all three, matching the explicit diagonal blocks
    of the Coulson effective Fock. This is not the separate spin occupied/
    virtual semicanonicalization used in Knowles RMP2 denominators.
    """
    c = np.array(coefficients, dtype=complex, copy=True)
    _counts(n_alpha, n_beta, c.shape[-1])
    eps = np.empty((len(c), c.shape[-1]))
    fc = .5*(np.asarray(fock_alpha)+np.asarray(fock_beta))
    for k in range(len(c)):
        for lo, hi in ((0, n_beta), (n_beta, n_alpha), (n_alpha, c.shape[-1])):
            block = c[k, :, lo:hi]
            e, u = np.linalg.eigh(_hermitian(block.conj().T @ fc[k] @ block))
            c[k, :, lo:hi] = block @ u
            eps[k, lo:hi] = e
    return c, eps


def _physical_state(c, h, s, weights, operator, n_alpha, n_beta, madelung, e_nuclear):
    da, db = rohf_densities(c, n_alpha, n_beta)
    j = np.asarray(operator.coulomb(da+db))
    ka = np.asarray(operator.exchange(da)) + madelung*(s @ da @ s)
    kb = np.asarray(operator.exchange(db)) + madelung*(s @ db @ s)
    fa, fb = _hermitian(h+j-ka), _hermitian(h+j-kb)
    def trace(d, a):
        return float(np.einsum('k,kpq,kqp->', weights, d, a).real)
    ej = .5*trace(da+db, j)
    ek = -.5*(trace(da, ka)+trace(db, kb))
    ee = trace(da+db, h)+ej+ek
    return dict(energy=ee+e_nuclear, e_electronic=ee, e_coulomb=ej,
                e_hf_exchange=ek, density_alpha=da, density_beta=db,
                density=da+db, fock_alpha=fa, fock_beta=fb)


class _RotationChart:
    """Complex inter-category exponential coordinates and exact chain rule."""
    def __init__(self, coefficients, n_alpha, n_beta):
        self.c = np.asarray(coefficients)
        self.nk, _, self.nmo = self.c.shape
        categories = np.where(np.arange(self.nmo) < n_beta, 0,
                             np.where(np.arange(self.nmo) < n_alpha, 1, 2))
        self.p, self.q = np.nonzero(np.tril(categories[:, None] != categories[None, :], -1))
        self.size = 2*self.nk*len(self.p)

    def rotate(self, parameters):
        z = np.asarray(parameters).reshape(self.nk, 2, -1)
        k = np.zeros((self.nk, self.nmo, self.nmo), complex)
        k[:, self.p, self.q] = z[:, 0]+1j*z[:, 1]
        k[:, self.q, self.p] = -z[:, 0]+1j*z[:, 1]
        theta, v = np.linalg.eigh(-1j*k)
        e = np.exp(1j*theta)
        u = (v*e[:, None, :]) @ v.conj().transpose(0, 2, 1)
        return self.c @ u, (theta, v, e, u)

    def pack_gradient(self, gradient, weights):
        lower = gradient[:, self.p, self.q]
        return (2*np.stack((lower.real, lower.imag), axis=1)*np.asarray(weights)[:,None,None]).ravel()

    def pullback(self, state, cache, n_alpha, n_beta, weights):
        theta, v, e, u = cache
        # dE/dU in the fixed chart basis; derivative of the matrix exponential
        # uses its divided differences, including the repeated-eigenvalue limit.
        b = np.zeros_like(u)
        for key, count in (('fock_alpha', n_alpha), ('fock_beta', n_beta)):
            a = self.c.conj().transpose(0, 2, 1) @ state[key] @ self.c
            term = a @ u
            term[:, :, count:] = 0
            b += 2*term
        dtheta = theta[:, :, None]-theta[:, None, :]
        # exp divided difference = exp(i mean theta) * sinc(delta theta/2pi).
        divided = np.exp(.5j*(theta[:, :, None]+theta[:, None, :])) * np.sinc(dtheta/(2*np.pi))
        g = v @ (divided.conj()*(v.conj().transpose(0, 2, 1) @ b @ v)) @ v.conj().transpose(0, 2, 1)
        grad = np.stack(((g[:, self.p, self.q]-g[:, self.q, self.p]).real,
                         (g[:, self.p, self.q]+g[:, self.q, self.p]).imag), axis=1)
        return (grad*np.asarray(weights)[:, None, None]).ravel()


def _rohf_gradient_response(chart, direction, state, overlap, weights, operator,
                            n_alpha, n_beta, madelung):
    """Exact linear response of the moving-frame ROHF gradient.

    Differentiate Guest--Saunders Eq. (12): dD_sigma=C[K,n_sigma]C^H,
    dF_sigma=J[dDa+dDb]-K[dD_sigma], and dFmo=C^H dF C+[Fmo,K].
    No numerical Hessian differences or perturbed native integrals are used.
    """
    z = np.asarray(direction).reshape(chart.nk, 2, -1)
    k = np.zeros((chart.nk, chart.nmo, chart.nmo), complex)
    k[:,chart.p,chart.q] = z[:,0]+1j*z[:,1]
    k[:,chart.q,chart.p] = -z[:,0]+1j*z[:,1]
    c, s = chart.c, overlap
    occupation = [(np.arange(chart.nmo) < count).astype(float) for count in (n_alpha,n_beta)]
    changes = [c @ (k*(o[None,None,:]-o[None,:,None])) @ c.conj().transpose(0,2,1)
               for o in occupation]
    j = operator.coulomb(changes[0]+changes[1])
    response = np.zeros_like(k)
    for spin, suffix in enumerate(('alpha','beta')):
        dd, o = changes[spin], occupation[spin]
        df = j-operator.exchange(dd)-madelung*(s @ dd @ s)
        fm = c.conj().transpose(0,2,1) @ state['fock_'+suffix] @ c
        dfm = c.conj().transpose(0,2,1) @ df @ c+fm@k-k@fm
        response += dfm*(o[None,None,:]-o[None,:,None])
    return chart.pack_gradient(response, weights)


def optimize_rohf(hcore, overlap, weights, operator, n_alpha, n_beta, *,
                  e_nuclear=0., madelung=0., max_iter=128,
                  energy_tolerance=1e-11, gradient_tolerance=1e-9,
                  linear_dep_threshold=1e-8, initial_coefficients=None,
                  memory_byte_cap=512*1024**2, progress=None):
    """Optimize one shared spatial frame and rebuild its physical final state.

    L-BFGS uses exact complex exponential-coordinate derivatives. Resetting
    the local chart avoids treating a singular parameter gradient as orbital
    stationarity. Analytic gradient-response Newton steps resolve residuals
    below the energy line-search precision floor. The final verdict always
    uses the three physical blocks.
    """
    from scipy.optimize import minimize
    from ...rohf import roothaan_effective_fock, require_nonnegative_max_iter
    require_nonnegative_max_iter(max_iter, route='chi ROHF')
    h, s, w = np.asarray(hcore), np.asarray(overlap), np.asarray(weights)
    if (h.ndim != 3 or h.shape != s.shape or h.shape[1] != h.shape[2]
            or not all(np.isfinite(a).all() for a in (h, s, w))
            or w.shape != (len(h),) or np.any(w <= 0) or abs(w.sum()-1) > 1e-12
            or max(np.max(abs(h-h.conj().transpose(0,2,1))),
                   np.max(abs(s-s.conj().transpose(0,2,1)))) > 1e-10):
        raise ValueError('ROHF requires finite Hermitian hcore/overlap and positive normalized weights')
    if not all(np.isfinite(t) and t > 0 for t in (energy_tolerance, gradient_tolerance, linear_dep_threshold)):
        raise ValueError('ROHF tolerances must be positive and finite')
    frames = []
    for hk, sk in zip(h, s):
        diagonal = sk.diagonal().real
        if np.any(diagonal <= 0):
            raise ValueError('ROHF overlap requires positive AO norms')
        scale = diagonal**-.5
        ev, vectors = np.linalg.eigh(sk*scale[:,None]*scale[None,:])
        if np.min(ev) < -linear_dep_threshold:
            raise ValueError('ROHF overlap has a negative metric direction')
        keep = ev >= linear_dep_threshold
        if not np.any(keep):
            raise ValueError('ROHF overlap retained rank is zero')
        x = scale[:,None]*vectors[:,keep]/np.sqrt(ev[keep])
        _, u = np.linalg.eigh(_hermitian(x.conj().T @ hk @ x))
        frames.append(x @ u)
    if len({c.shape[-1] for c in frames}) != 1:
        raise NotImplementedError('ROHF requires equal retained spatial rank at every character')
    c = np.asarray(frames)
    _counts(n_alpha, n_beta, c.shape[-1])
    if initial_coefficients is not None:
        c = np.asarray(initial_coefficients, complex).copy()
        if c.shape != np.asarray(frames).shape or not np.isfinite(c).all():
            raise ValueError('initial ROHF coefficients have incompatible retained rank')
        if np.max(abs(c.conj().transpose(0,2,1) @ s @ c-np.eye(c.shape[-1]))) > 1e-9:
            raise ValueError('initial ROHF coefficients are not metric orthonormal')
    chart = _RotationChart(c, n_alpha, n_beta)
    # Dense per-character matrix exponentials plus L-BFGS history; factors are
    # borrowed, and external BLAS/interpreter memory is outside this estimate.
    if 512*len(c)*c.shape[-1]**2+256*chart.size > memory_byte_cap:
        raise MemoryError('ROHF orbital optimization exceeds its numerical workspace budget')
    def physical(coeff):
        return _physical_state(coeff, h, s, w, operator, n_alpha, n_beta, madelung, e_nuclear)
    state = physical(c)
    trace = []
    previous_energy = state['energy']
    last_delta = 0.
    remaining = max_iter
    while remaining > 0 and chart.size:
        chart = _RotationChart(c, n_alpha, n_beta)
        residuals = rohf_residuals(c, state['fock_alpha'], state['fock_beta'], n_alpha, n_beta, w)
        if gradient_tolerance < max(residuals.values()) < 1e-5:
            # At machine-energy precision an energy line search cannot resolve
            # the remaining orbital motion. Solve the physical gradient's
            # analytic linear response and demand a smaller physical residual.
            from scipy.sparse.linalg import LinearOperator, gmres
            g = chart.pack_gradient(rohf_gradient(c, state['fock_alpha'],
                                    state['fock_beta'], n_alpha, n_beta), w)
            action = LinearOperator((chart.size,chart.size), dtype=float,
                matvec=lambda v: _rohf_gradient_response(chart,v,state,s,w,operator,n_alpha,n_beta,madelung))
            step, _ = gmres(action,-g,rtol=.01,atol=0.,restart=min(20,chart.size),maxiter=3)
            if np.isfinite(step).all() and np.linalg.norm(step) < .2:
                for factor in (1., .5, .25, .125):
                    trial, _ = chart.rotate(step*factor)
                    candidate = physical(trial)
                    rg = rohf_residuals(trial,candidate['fock_alpha'],candidate['fock_beta'],n_alpha,n_beta,w)
                    if (max(rg.values()) < max(residuals.values()) and
                            candidate['energy'] <= state['energy']+energy_tolerance):
                        last_delta = candidate['energy']-state['energy']
                        c, state = trial, candidate
                        previous_energy = state['energy']
                        remaining -= 1
                        trace.append(SimpleNamespace(iter=len(trace)+1,energy=state['energy'],
                            delta_e=last_delta,grad_norm=rg['generalized_rohf_gradient_norm'],diis_subspace=0))
                        if progress is not None:
                            progress.iteration(len(trace),energy=state['energy'],dE=last_delta,
                                grad=rg['generalized_rohf_gradient_norm'],diis=0)
                        break
                else:
                    candidate = None
                if candidate is state:
                    if max(rg.values()) <= gradient_tolerance and abs(last_delta) <= energy_tolerance:
                        break
                    continue
        last = {}
        def objective(parameters):
            cc, cache = chart.rotate(parameters)
            ss = physical(cc)
            last.update(parameters=np.array(parameters), c=cc, state=ss)
            return ss['energy'], chart.pullback(ss, cache, n_alpha, n_beta, w)
        def callback(parameters):
            nonlocal previous_energy, last_delta
            if not np.array_equal(parameters, last.get('parameters')):
                objective(parameters)
            ss = last['state']
            last_delta = ss['energy']-previous_energy
            previous_energy = ss['energy']
            grad = rohf_residuals(last['c'], ss['fock_alpha'], ss['fock_beta'], n_alpha, n_beta, w)
            trace.append(SimpleNamespace(iter=len(trace)+1, energy=ss['energy'],
                delta_e=last_delta, grad_norm=grad['generalized_rohf_gradient_norm'], diis_subspace=0))
            if progress is not None:
                progress.iteration(len(trace), energy=ss['energy'], dE=last_delta,
                                   grad=grad['generalized_rohf_gradient_norm'], diis=0)
        solution = minimize(objective, np.zeros(chart.size), jac=True, method='L-BFGS-B',
            callback=callback, options=dict(maxiter=min(32, remaining), maxls=30, maxcor=8,
                ftol=np.finfo(float).eps, gtol=gradient_tolerance/(8*max(1, len(c)))))
        c, _ = chart.rotate(solution.x)
        state = physical(c)
        remaining -= max(1, int(solution.nit))
        residuals = rohf_residuals(c, state['fock_alpha'], state['fock_beta'], n_alpha, n_beta, w)
        if max(residuals.values()) <= gradient_tolerance and abs(last_delta) <= energy_tolerance:
            break
    # Post-optimization within-category canonicalization cannot change either
    # spin density. Rebuild J/K and energy from the actual reported orbitals.
    old_energy = state['energy']
    old_da, old_db = state['density_alpha'], state['density_beta']
    c, eps = canonicalize_rohf(c, state['fock_alpha'], state['fock_beta'], n_alpha, n_beta)
    state = physical(c)
    residuals = rohf_residuals(c, state['fock_alpha'], state['fock_beta'], n_alpha, n_beta, w)
    metric_error = float(np.max(abs(c.conj().transpose(0,2,1) @ s @ c-np.eye(c.shape[-1]))))
    density_error = float(max(np.max(abs(state['density_alpha']-old_da)), np.max(abs(state['density_beta']-old_db))))
    electron_errors = [abs(float(np.einsum('k,kpq,kqp->', w, state[key], s).real)-target)
                       for key, target in (('density_alpha', n_alpha), ('density_beta', n_beta))]
    accepted = bool(max_iter > 0 and max(residuals.values()) <= gradient_tolerance
                    and abs(last_delta) <= energy_tolerance and metric_error <= 1e-9
                    and density_error <= 1e-9 and max(electron_errors) <= 1e-9
                    and abs(state['energy']-old_energy) <= energy_tolerance)
    report = dict(final_rebuild=True, stationary=accepted, accepted=accepted,
                  energy_residual=abs(last_delta), energy_consistency_error=abs(state['energy']-old_energy),
                  energy_tolerance=energy_tolerance, orbital_tolerance=gradient_tolerance,
                  orbital_residual=residuals['generalized_rohf_gradient_norm'],
                  density_refill_residual=density_error, metric_orthogonality_error=metric_error,
                  electron_count_errors=electron_errors, stability='not-assessed', **residuals)
    if not trace:
        trace.append(SimpleNamespace(iter=0, energy=state['energy'], delta_e=0.,
                     grad_norm=residuals['generalized_rohf_gradient_norm'], diis_subspace=0))
    occ_a = np.tile((np.arange(c.shape[-1]) < n_alpha).astype(float), (len(c),1))
    occ_b = np.tile((np.arange(c.shape[-1]) < n_beta).astype(float), (len(c),1))
    effective = np.asarray([roothaan_effective_fock(fa, fb, da, db, sk)
                           for fa, fb, da, db, sk in zip(state['fock_alpha'], state['fock_beta'],
                               state['density_alpha'], state['density_beta'], s)])
    spin = .5*(n_alpha-n_beta)
    return SimpleNamespace(**state, e_nuclear=float(e_nuclear), e_xc=0., converged=accepted,
        n_iter=trace[-1].iter, scf_trace=trace, overlap=s.copy(), hcore=h.copy(),
        mo_coeffs=c, mo_coeffs_alpha=c.copy(), mo_coeffs_beta=c.copy(), mo_energies=eps,
        mo_energies_alpha=np.real(np.diagonal(c.conj().transpose(0,2,1) @ state['fock_alpha'] @ c, axis1=1, axis2=2)),
        mo_energies_beta=np.real(np.diagonal(c.conj().transpose(0,2,1) @ state['fock_beta'] @ c, axis1=1, axis2=2)),
        occupations=occ_a+occ_b, mo_occupations=occ_a+occ_b,
        occupations_alpha=occ_a, occupations_beta=occ_b, fock=effective,
        n_alpha=n_alpha, n_beta=n_beta, n_closed=n_beta, n_open=n_alpha-n_beta,
        rohf_subspaces=_subspaces(c.shape[-1], n_alpha, n_beta),
        rohf_convention=ROHF_CONVENTION, rohf_canonicalization=ROHF_CANONICALIZATION,
        s_squared=spin*(spin+1), s_squared_ideal=spin*(spin+1),
        chi_scf_acceptance=report, termination_reason='converged' if accepted else 'iteration_limit',
        stability_status='not-assessed', smearing_temperature=0., kpoint_weights=w.copy(),
        accepted_density_k=(state['density_alpha'].copy(), state['density_beta'].copy()),
        returned_operator='iterated', energy_operator='iterated', ecp_total_ncore=0, method='rohf')


def validate_rohf_state(result):
    """Recheck the shared orbital/subspace contract independently of its flag."""
    required = ('mo_coeffs', 'overlap', 'n_alpha', 'n_beta', 'n_closed', 'n_open',
                'rohf_subspaces', 'rohf_convention', 'rohf_canonicalization',
                'fock_alpha', 'fock_beta', 'density_alpha', 'density_beta',
                'mo_coeffs_alpha', 'mo_coeffs_beta', 'occupations_alpha',
                'occupations_beta', 'occupations', 'mo_energies', 'kpoint_weights')
    if any(not hasattr(result, name) for name in required):
        raise ValueError('ROHF reference lacks its shared spatial/subspace state')
    c, s = np.asarray(result.mo_coeffs), np.asarray(result.overlap)
    if c.ndim != 3 or s.shape != (len(c), c.shape[1], c.shape[1]):
        raise ValueError('ROHF reference has incompatible spatial/metric shapes')
    na, nb = result.n_alpha, result.n_beta
    _counts(na, nb, c.shape[-1])
    if (result.n_closed != nb or result.n_open != na-nb
            or result.rohf_subspaces != _subspaces(c.shape[-1], na, nb)
            or result.rohf_convention != ROHF_CONVENTION
            or result.rohf_canonicalization != ROHF_CANONICALIZATION):
        raise ValueError('ROHF subspace/canonicalization identity mismatch')
    arrays = [c, s, result.fock_alpha, result.fock_beta, result.density_alpha,
              result.density_beta, result.occupations, result.mo_energies]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite ROHF reference')
    da, db = rohf_densities(c, na, nb)
    errors = [np.max(abs(c.conj().transpose(0,2,1) @ s @ c-np.eye(c.shape[-1]))),
              np.max(abs(da-result.density_alpha)), np.max(abs(db-result.density_beta))]
    for suffix, count in (('alpha', na), ('beta', nb)):
        expected = np.tile((np.arange(c.shape[-1]) < count).astype(float), (len(c),1))
        errors.extend([np.max(abs(getattr(result, 'occupations_'+suffix)-expected)),
                       np.max(abs(getattr(result, 'mo_coeffs_'+suffix)-c))])
    expected = (np.arange(c.shape[-1]) < na).astype(int)+(np.arange(c.shape[-1]) < nb).astype(int)
    errors.append(np.max(abs(result.occupations-expected)))
    residuals = rohf_residuals(c, result.fock_alpha, result.fock_beta, na, nb, result.kpoint_weights)
    tol = min(1e-9, float(result.chi_scf_acceptance.get('orbital_tolerance', 1e-9)))
    if max(errors) > 1e-9 or max(residuals.values()) > tol:
        raise ValueError('ROHF reference fails shared-spatial generalized stationarity')
    return residuals


def require_rohf_controls(*, dimension, backend, fit_source, controls):
    """Public early admission; no integral work and no inherited SR capability."""
    from .correlation import require_public_mp2_controls
    if dimension != 3 or backend != 'ri' or fit_source != 'classic':
        raise NotImplementedError('chi ROHF requires 3D classic physical-pair RI full Coulomb')
    require_public_mp2_controls(fit_source, controls)


def _build_source(system, basis, kpoints, options, controls, auxiliary_name):
    from ..._vibeqc_core import compute_overlap_lattice, compute_kinetic_lattice, bloch_sum
    from ...aux_basis import default_aux_for, make_aux_basis_set, make_modrho_aux_basis
    from ...periodic_fitted_support import (build_chi_fitted_operator,
        fitted_one_electron_options, preflight_fitted_one_electron,
        fitted_nuclear_operators, fitted_one_electron_record)
    from .ecp_hamiltonian import require_ecp_support, build_ecp_components, bloch_ecp_components, record_ecp_support
    contract = require_ecp_support(system, options, controls, 'classic', basis=basis)
    from ...periodic_k_gdf import _periodic_ecp_context
    context = _periodic_ecp_context(options, system, 'chi ROHF') if contract is not None else None
    ionic_system = context[4] if context is not None else system
    opts = fitted_one_electron_options(options.lattice_opts, controls)
    reserved = preflight_fitted_one_electron(system, basis, opts, controls,
        n_kpoints=len(kpoints.weights), ecp=contract is not None)
    slat = compute_overlap_lattice(basis, system, opts)
    tlat = compute_kinetic_lattice(basis, system, opts)
    vlat, reciprocal, nuclear = fitted_nuclear_operators(ionic_system, basis, opts, slat, controls)
    ecp_lat, ecp_record = build_ecp_components(basis, system, opts, context, contract) if contract is not None else ({}, None)
    record = fitted_one_electron_record(ionic_system, basis, opts,
        dict(S=slat,T=tlat,V_ne=vlat,**ecp_lat), reciprocal, nuclear, controls)
    if ecp_record is not None:
        record = record_ecp_support(record, ecp_record, controls)
    enuc = float(nuclear.energy)
    del reciprocal, nuclear
    s = np.asarray([_hermitian(np.asarray(bloch_sum(slat, k))) for k in kpoints.kpoints_cart])
    h = np.asarray([_hermitian(np.asarray(bloch_sum(tlat,k))+np.asarray(bloch_sum(vlat,k))) for k in kpoints.kpoints_cart])
    h_components = None
    if ecp_record is not None:
        h_components = bloch_ecp_components(dict(kinetic=tlat, ionic_coulomb=vlat, **ecp_lat), kpoints.kpoints_cart)
        h = sum(h_components.values())
    name = auxiliary_name or default_aux_for(basis.name)
    mol = system.unit_cell_molecule()
    aux = make_modrho_aux_basis(make_aux_basis_set(mol, aux_name=name), mol)
    operator = build_chi_fitted_operator(system, basis, aux, kpoints.kpoints_cart, kpoints.weights,
                                         source='classic', controls=controls, _retained_bytes=reserved)
    built = (h, s, enuc, operator, record, aux, name, opts)
    return built + (ecp_record, h_components) if ecp_record is not None else built


def run_chi_rohf(system, basis, lattice_extension=None, options=None, *, mesh=None,
        wigner_seitz_shells=None, backend='four_center', aux_basis=None, gdf_method='rsgdf',
        fit_source='auto', support_policy=None, coulomb_controls=None, hf_reference=None,
        initial_density_k=None, gdf_linear_dep_threshold=1e-9, rsgdf_ke_cutoff=200.,
        rsgdf_tail_ke_cutoff=None, mdf_ke_cutoff=40., fock_mixing=None, symmetry_mode='off',
        symmetry_precision=1e-5, symmetry_require_full_group=False, progress=None, verbose=None):
    from . import scf
    from .reference import _hf_request_identity, ChiHFReference
    from ..._vibeqc_core import PeriodicRHFOptions
    from ...periodic_k_gdf import _madelung_for_kmesh
    from ...periodic_fit_source import attach_fit_source
    from ...banner import loaded_process_provenance
    from ...progress import resolve_progress
    from ..exchange_convention import BVK_EWALD
    scf._warn_experimental()
    opts = options if options is not None else PeriodicRHFOptions()
    if type(opts) is not PeriodicRHFOptions:
        raise TypeError('chi ROHF requires PeriodicRHFOptions; KS/hybrid states are unsupported')
    selected = scf._resolve_backend(backend)
    require_rohf_controls(dimension=int(system.dim), backend=selected.value,
                          fit_source=fit_source, controls=coulomb_controls)
    scf._validate_open_shell_problem(system, opts)
    from .ecp_hamiltonian import require_ecp_support, attach_ecp_result
    ecp_request = require_ecp_support(system, opts, coulomb_controls, fit_source, basis=basis)
    if gdf_method != 'rsgdf':
        raise NotImplementedError('chi ROHF requires the explicit classic rsgdf source builder')
    if initial_density_k is not None:
        raise NotImplementedError('chi ROHF density restarts require a shared spatial frame and are unsupported')
    mixing = scf.resolve_fock_mixing(opts, fock_mixing, where='chi ROHF')
    if mixing != 0 or float(getattr(opts, 'level_shift', 0.)) != 0 or list(getattr(opts,'level_shift_schedule',()) or ()):
        raise NotImplementedError('chi ROHF orbital minimization requires zero Fock mixing and level shift')
    guess = getattr(getattr(opts, 'initial_guess', 'AUTO'), 'name', str(getattr(opts, 'initial_guess', 'AUTO')).split('.')[-1]).upper()
    if guess not in ('AUTO', 'HCORE', 'CORE'):
        raise NotImplementedError('chi ROHF orbital minimization currently requires AUTO/HCORE shared orbitals')
    scf._require_inactive_quadratic_fallback(opts, where='chi ROHF')
    _, _, cutoff, tail, threshold = scf._hf_support_contract(system, opts, selected, fit_source,
        gdf_method, support_policy, coulomb_controls, rsgdf_ke_cutoff, rsgdf_tail_ke_cutoff, gdf_linear_dep_threshold)
    extension = scf._resolve_lattice_extension(system, lattice_extension, mesh=mesh, wigner_seitz_shells=wigner_seitz_shells)
    kpoints = scf.cyclic_gamma_mesh(system, extension)
    if scf._resolve_symmetry_mode(symmetry_mode) is not scf.AICCM2026DevBSymmetryMode.OFF:
        raise NotImplementedError('chi ROHF requires unreduced characters with symmetry_mode=off')
    plan = None
    controls = dict(backend=selected.value, aux_basis=aux_basis, gdf_method=gdf_method,
        fit_source=fit_source, support_policy=support_policy, coulomb_controls=coulomb_controls,
        rsgdf_ke_cutoff=cutoff, rsgdf_tail_ke_cutoff=tail, gdf_linear_dep_threshold=threshold,
        mdf_ke_cutoff=mdf_ke_cutoff, fock_mixing=mixing, initial_density_k=initial_density_k,
        symmetry_mode=str(symmetry_mode), symmetry_precision=symmetry_precision,
        symmetry_require_full_group=symmetry_require_full_group)
    identity = _hf_request_identity(system, basis, extension, opts, 'ROHF', controls)
    if hf_reference is not None:
        if not isinstance(hf_reference, ChiHFReference):
            raise TypeError('hf_reference must be prepared by prepare_chi_hf_reference')
        return hf_reference.validate(identity, 'ROHF')
    plog = resolve_progress(progress, verbose=verbose)
    started = perf_counter()
    built = _build_source(system, basis, kpoints, opts, coulomb_controls, aux_basis)
    h, s, enuc, operator, record, aux, name, lat = built[:8]
    ecp_record, h_components = built[8:] if len(built) == 10 else (None, None)
    if (ecp_request is None) != (ecp_record is None):
        raise ValueError('chi ROHF source did not execute its requested ECP Hamiltonian')
    ne = int(ecp_request['explicit_electrons']) if ecp_request is not None else int(system.n_electrons())
    spin = int(system.multiplicity)-1
    result = optimize_rohf(h, s, kpoints.weights, operator, (ne+spin)//2, (ne-spin)//2,
        e_nuclear=enuc, madelung=_madelung_for_kmesh(system, extension), max_iter=opts.max_iter,
        energy_tolerance=opts.conv_tol_energy, gradient_tolerance=opts.conv_tol_grad,
        linear_dep_threshold=1e-8, memory_byte_cap=coulomb_controls.memory_byte_cap, progress=plog)
    # Match the established multi-character result presentation. In particular
    # Gamma orbital exporters select a character from lists of AO/MO arrays.
    for key in ('mo_coeffs', 'mo_coeffs_alpha', 'mo_coeffs_beta', 'mo_energies',
                'mo_energies_alpha', 'mo_energies_beta', 'occupations',
                'occupations_alpha', 'occupations_beta', 'mo_occupations',
                'overlap', 'hcore', 'fock', 'fock_alpha', 'fock_beta',
                'density', 'density_alpha', 'density_beta'):
        setattr(result, key, [np.asarray(row).copy() for row in getattr(result, key)])
    result.accepted_density_k = (result.density_alpha, result.density_beta)
    result.backend = 'chi-classic-shared-spatial-rohf'
    result.exchange_q0 = BVK_EWALD
    result.kpoints_cart = np.asarray(kpoints.kpoints_cart).copy()
    attach_ecp_result(result, ecp_record, h_components)
    result.aux_basis_name, result.n_aux = name, int(aux.nbasis)
    result.process_provenance = loaded_process_provenance()
    result.fock_mixing = result.level_shift = 0.
    result.level_shift_warmup_cycles = 0
    result._chi_hf_request_identity = identity
    from .checkpoint import bind_chi_restart_inputs
    bind_chi_restart_inputs(result, system, basis, extension, opts, 'ROHF', controls)
    result.chi_scf_controls = dict(optimizer='shared-spatial-complex-exponential-L-BFGS/Newton',
        requested_fock_mixing=mixing, executed_fock_mixing=0., requested_level_shift=0.,
        executed_level_shift_warmup_cycles=0, requested_damping=float(getattr(opts,'damping',0.)),
        requested_dynamic_damping=bool(getattr(opts,'dynamic_damping',False)),
        requested_use_diis=bool(getattr(opts,'use_diis',False)),
        requested_scf_accelerator=scf._scf_accelerator_name(opts),
        executed_use_diis=False, executed_dynamic_damping=False,
        executed_density_damping=0., requested_initial_guess=guess, executed_initial_guess='HCORE',
        final_operator='rebuilt-unshifted-unmixed', stability='not-assessed', method_maturity='experimental')
    attach_fit_source(result, system=system, basis=basis, auxiliary=aux, auxiliary_name=name,
        cache=operator, request=None, selected='classic', lat_opts=lat, oneel_lat_opts=lat,
        metric_threshold=threshold, fit_screen_threshold=0., kpoints_cart=kpoints.kpoints_cart,
        executed_tail=None, retain_canonical=True, one_electron_record=record,
        classic_support_policy='physical_pairs')
    scf._attach_diagnostics(result, system, basis, kpoints, extension, selected, 'ROHF',
        scf._resolve_exact_exchange_coefficients(None, spin=2, where='chi ROHF'),
        perf_counter()-started, plan, opts, executed_level_shift_warmup_cycles=0)
    from dataclasses import replace
    result.aiccm2026dev_b = replace(result.aiccm2026dev_b, use_diis=False,
        scf_accelerator='shared-spatial-complex-exponential-L-BFGS/Newton',
        damping=0., dynamic_damping=False)
    scf._attach_fitted_selector_settings(result, selected, gdf_method=gdf_method,
        rsgdf_ke_cutoff=cutoff, rsgdf_tail_ke_cutoff=None, mdf_ke_cutoff=mdf_ke_cutoff)
    return result
