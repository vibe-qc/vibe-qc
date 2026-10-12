"""Public experimental MP2 on explicit, retained chi RI HF sources.

Admission describes a numerical capability, not production qualification.
No environment flag, refit or CCM producer can replace the executed source.
"""
from dataclasses import dataclass, replace
import json
import math
import numbers


PUBLIC_MP2_ADMISSION = 'admitted: experimental chi same-source RI MP2/UMP2 v1'
_PUBLIC_MP2_ROUTES = frozenset({('RHF', 'classic'), ('RHF', 'sr'), ('UHF', 'classic')})


def require_public_mp2_admission(correlation, method, fit_source):
    """Admit only the explicit source and HF spin capabilities of R5.1."""
    if correlation != 'mp2' or method not in ('RHF', 'UHF'):
        raise NotImplementedError(
            'chi same-source correlation supports only HF MP2/UMP2; '
            'KS, ROHF and CCSD(T) are unsupported')
    if fit_source not in ('classic', 'sr'):
        raise NotImplementedError(
            'chi same-source MP2 requires explicit classic or sr; '
            'historical auto defaults are unchanged. '
            'No SCF-only result is returned for requested correlation.')
    if (method, fit_source) not in _PUBLIC_MP2_ROUTES:
        raise NotImplementedError(
            'public chi SR UMP2 remains gated pending independent native '
            'complex-spin operator acceptance; no SCF-only correlated success')
    return PUBLIC_MP2_ADMISSION


def require_public_mp2_controls(fit_source, controls):
    """Keep public correlation within the explicit finite-source contract."""
    from .fitted import ChiCoulombControls
    if controls is None:
        raise NotImplementedError('chi MP2 requires explicit chi_coulomb_controls=ChiCoulombControls(...)')
    if not isinstance(controls, ChiCoulombControls):
        raise TypeError('chi MP2 requires explicit chi_coulomb_controls=ChiCoulombControls(...)')
    controls.require_source(fit_source)
    if (controls.support_policy != 'physical_pairs'
            or controls.tail_reciprocal_cutoff_ha is not None
            or controls.integral_screen_error != 0
            or controls.fit_screen_threshold != 0):
        raise NotImplementedError(
            'chi MP2 admits physical_pairs with no tail or integral/fit screening; '
            'historical support and other finite-source approximations remain gated')


def require_public_mp2_request(correlation, method, fit_source, *, dimension,
                               backend, controls, conv_tol_energy, conv_tol_grad):
    """Check known capabilities before SCF; achieved stationarity is checked later."""
    require_public_mp2_admission(correlation, method, fit_source)
    if dimension != 3 or backend != 'ri':
        raise NotImplementedError('chi MP2 requires 3D full-Coulomb RI HF (aiccm_backend="ri")')
    require_public_mp2_controls(fit_source, controls)
    for name, value, limit in (('conv_tol_energy', conv_tol_energy, 1e-11),
                               ('conv_tol_grad', conv_tol_grad, 1e-9)):
        if value is None:
            raise NotImplementedError(f'chi MP2 requires explicit 0 < {name} <= {limit:g}')
        if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value)) or value <= 0):
            raise ValueError(f'{name} must be a positive finite real')
        if value > limit:
            raise NotImplementedError(f'chi MP2 requires explicit 0 < {name} <= {limit:g}')


def require_public_mp2_mesh(lattice, mesh, kpoints_cart, weights):
    """Reject object-form meshes that would otherwise lose their sampling."""
    import numpy as np
    if mesh is None:
        raise NotImplementedError('chi MP2 requires a complete unshifted cyclic mesh')
    sizes = np.asarray(mesh)
    k = np.asarray(kpoints_cart, dtype=float)
    w = np.asarray(weights, dtype=float)
    if (sizes.shape != (3,) or not np.issubdtype(sizes.dtype, np.integer)
            or np.any(sizes < 1) or k.shape != (int(np.prod(sizes)), 3)
            or not np.isfinite(k).all() or w.shape != (len(k),)
            or not np.allclose(w, 1 / len(k), atol=1e-14, rtol=0)):
        raise NotImplementedError('chi MP2 requires complete uniform unreduced characters')
    scaled = (k @ np.asarray(lattice) / (2 * np.pi)) * sizes
    labels = np.rint(scaled)
    if (np.max(abs(scaled-labels)) > 1e-10
            or len(set(map(tuple, labels.astype(int) % sizes))) != len(k)):
        raise NotImplementedError('chi MP2 requires a complete unshifted cyclic mesh')


def validate_public_mp2_options(method, fit_source, frozen_core, auxiliary_batch_size):
    """Validate requested panel/core controls without running or altering HF."""
    from .canonical_mp2 import _positive_integer
    batch = _positive_integer(auxiliary_batch_size, 'chi_auxiliary_batch_size')
    if isinstance(frozen_core, numbers.Integral):
        frozen = (frozen_core, frozen_core)
    else:
        try:
            frozen = tuple(frozen_core)
        except TypeError as exc:
            raise ValueError('chi_frozen_core requires an integer or two spin counts') from exc
    if (len(frozen) != 2 or any(isinstance(n, bool)
            or not isinstance(n, numbers.Integral) or n < 0 for n in frozen)):
        raise ValueError('chi_frozen_core requires nonnegative integer spin counts')
    if method == 'RHF' and frozen[0] != frozen[1]:
        raise ValueError('restricted frozen cores must agree')
    if any(frozen) and (method, fit_source) != ('RHF', 'classic'):
        raise NotImplementedError(
            'public chi frozen-core MP2 is admitted only for explicit classic RHF; '
            'SR and UHF frozen active spaces await native acceptance')
    return tuple(int(n) for n in frozen), batch


@dataclass(frozen=True)
class ChiMP2JobResult:
    """Matched HF trace plus actual per-cell correlation and total energies.

    ``energy`` delegates to HF, as on the other periodic correlated runners.
    The runner replaces wrapper metadata without mutating the sealed HF state.
    """
    hf_result: object
    correlation: object
    guess_selection: object = None

    def __getattr__(self, name):
        return getattr(self.hf_result, name)

    @property
    def e_correlation(self):
        return self.correlation.e_corr_per_cell

    @property
    def e_total_correlated(self):
        return self.correlation.e_total_per_cell


def _evaluate_job(reference, *, frozen_core=0, auxiliary_batch_size=64):
    """Evaluate the retained source without refitting or rebuilding HF."""
    from .canonical_mp2 import evaluate_same_source_mp2
    correlation = evaluate_same_source_mp2(
        reference, frozen_core=frozen_core,
        auxiliary_batch_size=auxiliary_batch_size)
    return ChiMP2JobResult(reference.hf_result, correlation,
                           getattr(reference.hf_result, 'guess_selection', None))


def run_chi_mp2(reference, *, frozen_core=0, auxiliary_batch_size=64):
    """Run admitted experimental MP2/UMP2 on a sealed explicit RI HF state.

    The kernel rechecks the achieved HF residuals, integer gapped occupations,
    full character group, canonical metric and source identities. Public
    admission is recorded only after those checks and correlation succeed.
    """
    from .reference import ChiHFReference
    if not isinstance(reference, ChiHFReference):
        raise TypeError('chi MP2 requires a sealed ChiHFReference')
    hf = reference.validate(reference.request_identity, reference.method)
    source = getattr(getattr(hf, 'fitted_operator', None), 'descriptor', None)
    admission = require_public_mp2_admission(
        'mp2', reference.method, getattr(source, 'source', None))
    domains = json.loads(source.executed_domains_json)
    if (source.engine != 'native-chi-' + source.source + '-finite-contract'
            or domains.get('schema') != 'vibeqc.chi.fitted-domains/v1'
            or not isinstance(domains.get('controls'), dict)):
        raise ValueError('public chi MP2 requires the executed native finite-source contract')
    from .fitted import ChiCoulombControls
    require_public_mp2_controls(source.source, ChiCoulombControls(**domains['controls']))
    frozen, batch = validate_public_mp2_options(reference.method, source.source, frozen_core, auxiliary_batch_size)
    result = _evaluate_job(reference, frozen_core=frozen, auxiliary_batch_size=batch)
    return replace(result, correlation=replace(result.correlation, public_admission=admission))
