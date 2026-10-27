"""Explicit, domain-scoped empirical residuals for molecular RHF planning.

These profiles are predictive calibrations, not analytical allocation bounds.
The caller supplies authenticated runtime/harness identity and observations;
this module never discovers, downloads, or silently selects a profile.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math

_ACTIVE = ContextVar('runtime_memory_profile', default=None)
_IDENTITY = ('runtime_source', 'native_sha256', 'hardware_class', 'harness_sha256')
_SCHEMA = 'vibeqc.runtime-memory-baseline/1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('memory profile requires nonnegative integer byte counts and positive dimensions')
    return value


def fit_runtime_memory_profile(domain, observations):
    """Freeze the maximum positive training residual, with no fitted margin.

    Each observation is an independently measured fresh calculator process.
    ``allocation_bytes`` excludes the old assumed Python runtime floor. The
    residual absorbs unmodelled live allocations and allocator/library effects;
    it must not be interpreted as physically disjoint resident storage.
    """
    if not observations:
        raise ValueError('memory profile requires training observations')
    if set(domain) != set(_IDENTITY) | {'method', 'route', 'threads_per_rank', 'nbf'}:
        raise ValueError('memory profile requires an explicit runtime and dimension domain')
    if any(not isinstance(domain[k], str) or not domain[k] for k in _IDENTITY):
        raise ValueError('missing authenticated runtime/harness identity')
    if domain['method'] != 'rhf' or domain['route'] not in ('dense_conventional', 'direct_scf') or domain['threads_per_rank'] != 1:
        raise ValueError('only single-thread conventional/direct RHF profiles are supported')
    for key, length in (('runtime_source', 40), ('native_sha256', 64), ('harness_sha256', 64)):
        if len(domain[key]) != length or any(c not in '0123456789abcdef' for c in domain[key]):
            raise ValueError('invalid runtime or harness digest')
    if type(domain['threads_per_rank']) is not int:
        raise ValueError('thread count must be an integer')
    bounds = domain['nbf']
    if set(bounds) != {'min', 'max'} or _integer(bounds['min'], 1) > _integer(bounds['max'], 1):
        raise ValueError('invalid basis-count domain')
    seen = set()
    residuals = []
    for row in observations:
        if set(row) != {'observation_sha256', 'peak_bytes', 'allocation_bytes', 'nbf'}:
            raise ValueError('invalid training observation')
        digest = row['observation_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or digest in seen:
            raise ValueError('training observations require unique SHA256 digests')
        seen.add(digest)
        n = _integer(row['nbf'], 1)
        if not bounds['min'] <= n <= bounds['max']:
            raise ValueError('training count outside profile domain')
        residuals.append(max(0, _integer(row['peak_bytes'], 1) - _integer(row['allocation_bytes'])))
    if min(row['nbf'] for row in observations) != bounds['min'] or max(row['nbf'] for row in observations) != bounds['max']:
        raise ValueError('profile dimension bounds must equal observed training bounds')
    return {'schema': _SCHEMA, 'method': 'max_positive_training_residual',
            'scope': 'rank', 'value_bytes': max(residuals),
            'domain': json.loads(json.dumps(domain)), 'training': json.loads(json.dumps(observations))}


@contextmanager
def runtime_memory_profile(profile, identity):
    """Opt in to one frozen training profile and authenticated caller identity.

    The identity must describe the actual native/runtime and measured command
    harness. It is not an operator override for unknown hardware. A domain miss
    keeps the estimate unknown; malformed profiles are refused.
    """
    expected = fit_runtime_memory_profile(profile['domain'], profile['training'])
    if profile != expected:
        raise ValueError('memory profile differs from its reproducible training rule')
    token = _ACTIVE.set((expected, dict(identity)))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def apply_runtime_memory_profile(estimate, *, method, response=False):
    """Attach a scoped empirical prediction without changing legacy displays."""
    active = _ACTIVE.get()
    if active is None:
        return
    profile, identity = active
    block = estimate.analytical_memory
    features = estimate.resolved_features
    domain = profile['domain']
    terms = json.loads(block['terms_json'])
    terms = [t for t in terms if t['route'] != 'assumed_python_runtime_baseline']
    expected_names = set(estimate.by_category) - {'Python runtime + NumPy overhead'}
    match = (method == 'rhf' and not response and block['scope'] == 'rank'
             and all(identity.get(k) == domain[k] for k in _IDENTITY)
             and features.get('algorithm') == domain['route']
             and features.get('threads_per_rank') == domain['threads_per_rank']
             and type(features.get('nbf')) is int
             and domain['nbf']['min'] <= features['nbf'] <= domain['nbf']['max']
             and {t['name'] for t in terms} == expected_names
             and all(t['scope'] == 'rank' and t['route'] in
                     ('restricted_mean_field', domain['route']) for t in terms))
    if not match:
        unknown = json.loads(block['unknown_json'])
        unknown.append('empirical runtime profile outside authenticated domain or incomplete allocation coverage')
        block['unknown_json'] = json.dumps(unknown)
        return
    raw = sum(_integer(t['value_bytes']) for t in terms) + profile['value_bytes']
    if not math.isfinite(estimate.headroom_factor) or estimate.headroom_factor < 1:
        raise ValueError('invalid program memory headroom')
    block.update(complete=True, model_kind='analytical_allocations_with_empirical_residual',
                 terms_json=json.dumps(terms, sort_keys=True), unknown_json='[]',
                 raw_peak_bytes=raw, estimated_peak_bytes=int(raw * estimate.headroom_factor),
                 empirical_baseline_json=json.dumps({**profile, 'profile_sha256': _digest(profile)}, sort_keys=True))
