"""Independent vector-valued XC integration witnesses for benchmark fixtures.

Gaussian values and derivatives use the existing polynomial test oracle, not
the native AO evaluator. The functional integrand still comes from libxc.
Adaptive error estimates are numerical diagnostics, not rigorous bounds.
Nothing here is imported by the product runtime.
"""
import time

import numpy as np


def pointwise_xc(c, fractional_points):
    """Return volume-scaled [energy, spin populations, Re V, Im V] per point.

    Potential entries flatten in spin/k/AO-row/AO-column order. BZ weights enter
    the density, not each k potential. Density, functional and source stay fixed.
    """
    from tests.test_manta_periodic_semilocal_xc import bloch_values_gradients
    b, g = bloch_values_gradients(dict(c, points=fractional_points@c['lattice'].T))
    d, kw, f = c['densities'], c['weights'], c['functional']
    rho = np.einsum('k,kgi,skij,kgj->sg', kw, b, d, b.conj()).real
    grad = (np.einsum('k,akgi,skij,kgj->sag', kw, g, d, b.conj())
            + np.einsum('k,kgi,skij,akgj->sag', kw, b, d, g.conj())).real
    flow = np.zeros_like(grad)
    if len(d) == 1:
        energy, vr, vs = f.eval_unpolarised(rho[0], np.sum(grad[0]**2, axis=0))
        if len(vs): flow = 2*vs[None, None, :]*grad
        vr = vr[None, :]
    else:
        energy, va, vb, aa, ab, bb = f.eval_polarised(rho[0], rho[1],
            np.sum(grad[0]**2, axis=0), np.sum(grad[0]*grad[1], axis=0), np.sum(grad[1]**2, axis=0))
        vr = np.array([va, vb])
        if len(aa): flow = np.array([2*aa*grad[0]+ab*grad[1], 2*bb*grad[1]+ab*grad[0]])
    potential = np.einsum('kgi,sg,kgj->gskij', b.conj(), vr, b)
    potential += np.einsum('akgi,sag,kgj->gskij', g.conj(), flow, b)
    potential += np.einsum('kgi,sag,akgj->gskij', b.conj(), flow, g)
    potential = potential.reshape(len(fractional_points), -1)
    values = np.column_stack([energy, rho.T, potential.real, potential.imag])*abs(np.linalg.det(c['lattice']))
    if not np.isfinite(values).all(): raise ValueError('Nonfinite independent XC integrand')
    return values


def unpack_integral(values, density_shape):
    spins, entries = density_shape[0], int(np.prod(density_shape))
    start = 1+spins
    potential = (values[start:start+entries]+1j*values[start+entries:]).reshape(density_shape)
    return float(values[0]), potential, values[1:start]


def adaptive_reference(c, *, rule, atol, max_subdivisions):
    """Integrate every component with a capped independent subdivision rule.

    SciPy cubature implements product Gauss-Kronrod or degree-seven Genz-Malik
    (Genz and Malik, JCAM 6, 295, 1980; doi:10.1016/0771-050X(80)90039-X).
    Returns the actual status even when the subdivision cap is exhausted.
    """
    try:
        from scipy.integrate import cubature
    except ImportError as exc:
        raise RuntimeError('This optional reference requires scipy.integrate.cubature') from exc
    counts = dict(calls=0, points=0)
    def integrand(points):
        counts['calls'] += 1; counts['points'] += len(points)
        return pointwise_xc(c, points)
    start = time.perf_counter()
    result = cubature(integrand, np.zeros(3), np.ones(3), rule=rule,
        rtol=0, atol=atol, max_subdivisions=max_subdivisions, workers=1)
    energy, potential, populations = unpack_integral(result.estimate, c['densities'].shape)
    return dict(rule=rule, atol=atol, max_subdivisions=max_subdivisions,
        status=result.status, subdivisions=result.subdivisions, regions=len(result.regions),
        seconds=time.perf_counter()-start, counts=counts,
        max_estimated_component_error=float(np.max(result.error)), energy=energy,
        potential=potential, populations=populations)
