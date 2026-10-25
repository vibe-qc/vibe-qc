"""Portable, source-preserving restart for experimental classic chi HF.

The archive carries numeric state, never executable objects. Integrity is not
scientific acceptance: loading rebuilds the physical operator and verifies the
method's stationary projector before issuing a new process-local seal. Version
one has no implicit migrations and never invokes a source builder or SCF.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from ...output.checkpoint import read_restart_state, write_restart_state
from ...periodic_fit_source import RSGDFFitSource
from ...periodic_fitted_support import ChiFittedOperator
from .reference import ChiHFReference, _digest, _value

_SCHEMA = 'vibeqc.chi.hf-restart/v1'
_ECP_SCHEMA = 'vibeqc.chi.hf-restart/v2'
_ARRAY_FIELDS = ('overlap', 'hcore', 'mo_coeffs', 'mo_energies', 'occupations',
    'fock', 'density', 'mo_coeffs_alpha', 'mo_coeffs_beta', 'mo_energies_alpha',
    'mo_energies_beta', 'occupations_alpha', 'occupations_beta', 'fock_alpha',
    'fock_beta', 'density_alpha', 'density_beta', 'kpoints_cart', 'kpoint_weights',
    'mo_occupations')
_VALUE_FIELDS = ('energy', 'e_nuclear', 'e_electronic', 'e_coulomb', 'e_hf_exchange',
    'e_xc', 's_squared', 's_squared_ideal', 'n_alpha', 'n_beta', 'n_closed', 'n_open',
    'rohf_subspaces', 'rohf_convention', 'rohf_canonicalization', 'smearing_temperature',
    'ecp_total_ncore', 'returned_operator', 'energy_operator', 'backend', 'runtime_backend', 'aux_basis_name',
    'n_aux', 'n_iter', 'iterations', 'n_iterations', 'termination_reason', 'fock_mixing',
    'level_shift', 'level_shift_warmup_cycles', 'exchange_q0', 'chi_scf_controls',
    'finite_torus_convention', 'exact_exchange_assembly', 'executed_support',
    'aiccm2026dev_b', 'chi_scf_acceptance', 'linear_dependence',
    'chi_ecp', 'hcore_components', 'effective_n_electrons', 'ecp_effective_charges',
    'effective_nuclear_charge', 'effective_nuclear_charges')


@dataclass(frozen=True)
class _AttributeRecord(Mapping):
    """Allowlisted, inert metadata record; no serialized class is imported."""
    values: dict
    def __getitem__(self, key): return self.values[key]
    def __iter__(self): return iter(self.values)
    def __len__(self): return len(self.values)
    def as_dict(self): return _plain(self)
    def __getattr__(self, name):
        values = object.__getattribute__(self, 'values')
        if name in values:
            return values[name]
        if name in ('n_discarded_max','any_discarded') and 'n_kept_per_k' in values:
            discarded = int(values['n_basis'])-min(values['n_kept_per_k'],default=int(values['n_basis']))
            return discarded if name == 'n_discarded_max' else discarded > 0
        if name == 'condition_number' and 'min_overlap_eigenvalue' in values:
            low = float(values['min_overlap_eigenvalue']); high = float(values['max_overlap_eigenvalue'])
            return high/low if low > 0 else float('inf')
        convention = values.get('finite_torus_convention')
        if convention is not None:
            return getattr(convention, name)
        raise AttributeError(name)


@dataclass(frozen=True)
class ChiHFGuess:
    """Explicit orbital/density guess; never accepted for correlation."""
    method: str
    density_k: object
    mo_coeffs: object
    physical_identity: str
    convergence_required: bool = True
    restart_mode: str = 'guess'


def _plain(value):
    """Normalize public physics metadata, omitting transient execution fields."""
    if is_dataclass(value) and not isinstance(value, Mapping):
        value = {f.name: getattr(value, f.name) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()
                if k not in ('wall_time_seconds', 'wall_time_s', 'process_provenance',
                             'owner_pid', 'written_at', 'timestamp', 'runtime_backend')}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value
    if value is None or type(value) in (str, int, float, bool): return value
    raise TypeError(f'unsupported inert restart metadata: {type(value).__name__}')


def _state_arrays(hf, method):
    result = {}
    for name in _ARRAY_FIELDS:
        value = getattr(hf, name, None)
        if value is None or (hasattr(value, 'cells') and hasattr(value, 'blocks')):
            continue
        result[name] = np.asarray(value)
    accepted = getattr(hf, 'accepted_density_k', None)
    if accepted is not None:
        if method == 'RHF': result['density'] = np.asarray(accepted)
        else:
            result['density_alpha'], result['density_beta'] = map(np.asarray, accepted)
            result['density'] = result['density_alpha'] + result['density_beta']
    if method != 'RHF' and 'density_alpha' in result and 'density_beta' in result:
        result['density'] = result['density_alpha'] + result['density_beta']
    return result


def portable_reference_identity(reference):
    """Bind physical arrays/source/controls, excluding process and timing data."""
    if not isinstance(reference, ChiHFReference):
        raise TypeError('portable identity requires ChiHFReference')
    hf = reference.hf_result
    values = {name: _plain(getattr(hf, name)) for name in _VALUE_FIELDS
              if hasattr(hf, name) and name not in ('aiccm2026dev_b', 'chi_scf_acceptance',
                  'n_iter', 'iterations', 'n_iterations', 'termination_reason', 'runtime_backend')}
    operator = getattr(hf, 'fitted_operator', None)
    from ...periodic_fit_source import factor_digest
    return _digest(dict(method=reference.method, request=reference.request_identity,
        values=values, arrays=_state_arrays(hf, reference.method),
        physical=_plain(getattr(hf, '_chi_restart_physical', None)),
        source=_plain(getattr(hf, 'executed_fit_source', None)),
        operator=None if operator is None else operator.descriptor.as_dict(),
        factor_bytes=None if operator is None else factor_digest(operator)))


def _basis_record(basis):
    """Same normalized-shell identity as aux_basis, without loading native code."""
    digest = hashlib.sha256(b'vibeqc-basis-fingerprint-v1\0')
    shells = []
    for shell in basis.shells():
        header = (int(shell.atom_index), int(shell.l), int(bool(shell.pure)),
                  len(shell.exponents), len(shell.coefficients))
        digest.update(':'.join(str(v) for v in header).encode('ascii'))
        digest.update(b'\0')
        for values in (shell.exponents, shell.coefficients, shell.origin):
            a = np.asarray(values, dtype='<f8')
            digest.update(str(a.size).encode('ascii')); digest.update(b':'); digest.update(a.tobytes())
        shells.append(dict(atom_index=header[0], l=header[1], pure=bool(header[2]),
            exponents=list(map(float, shell.exponents)), coefficients=list(map(float, shell.coefficients)),
            origin=list(map(float, shell.origin))))
    return dict(name=str(basis.name), nbasis=int(basis.nbasis), sha256=digest.hexdigest(), shells=shells)


def _geometry_record(system):
    return dict(dimension=int(system.dim), lattice_bohr=np.asarray(system.lattice).tolist(),
        nuclei=[dict(Z=int(atom.Z), xyz_bohr=list(map(float, atom.xyz))) for atom in system.unit_cell],
        charge=int(system.charge), multiplicity=int(system.multiplicity),
        n_electrons=int(system.n_electrons()))


def bind_chi_restart_inputs(result_or_reference, system, basis, mesh, options, method, controls):
    """Bind complete portable request metadata while inputs are available."""
    hf = getattr(result_or_reference, 'hf_result', result_or_reference)
    from .ecp import ecp_contract
    ecp = ecp_contract(system, options)
    hf._chi_restart_physical = dict(geometry=_geometry_record(system), basis=_basis_record(basis),
        mesh=list(map(int, mesh)), method=str(method).upper(), options=_value(options), controls=_value(controls),
        ecp='all-electron' if ecp is None else ecp)
    return result_or_reference


def _encode(value, arrays):
    if isinstance(value, np.ndarray):
        key = 'state_' + str(len(arrays)); arrays[key] = value
        return {'array': key}
    if isinstance(value, np.generic): return _encode(value.item(), arrays)
    if isinstance(value, _AttributeRecord):
        return {'record': {k: _encode(v, arrays) for k, v in value.items()}}
    if is_dataclass(value):
        return {'record': {f.name: _encode(getattr(value, f.name), arrays) for f in fields(value)
                          if f.name not in ('wall_time_seconds', 'wall_time_s')}}
    if isinstance(value, Mapping): return {'mapping': {str(k): _encode(v, arrays) for k, v in value.items()}}
    if isinstance(value, tuple): return {'tuple': [_encode(v, arrays) for v in value]}
    if isinstance(value, list): return {'list': [_encode(v, arrays) for v in value]}
    if value is None or type(value) in (str, int, float, bool): return value
    raise TypeError(f'unsupported checkpoint value: {type(value).__name__}')


def _decode(value, arrays):
    if not isinstance(value, dict):
        if value is None or type(value) in (str, int, float, bool): return value
        raise ValueError('invalid restart value')
    if len(value) != 1: raise ValueError('invalid restart tagged value')
    tag, item = next(iter(value.items()))
    if tag == 'array':
        if item not in arrays: raise ValueError('missing restart array')
        return arrays[item]
    if tag in ('tuple', 'list') and isinstance(item, list):
        result = [_decode(v, arrays) for v in item]
        return tuple(result) if tag == 'tuple' else result
    if tag in ('mapping', 'record') and isinstance(item, dict):
        result = {k: _decode(v, arrays) for k, v in item.items()}
        return _AttributeRecord(result) if tag == 'record' else result
    raise ValueError('unknown restart value type; no executable reconstruction')


def _source(value):
    # A single explicitly allowed production type, with no dynamic import path.
    allowed = {f.name for f in fields(RSGDFFitSource)}
    if not isinstance(value, dict) or set(value) != allowed:
        raise ValueError('restart source descriptor fields mismatch')
    if value['schema'] != 'vibeqc.rsgdf.executed-source/v1' or value['engine_version'] != 1:
        raise ValueError('unsupported restart source descriptor schema')
    def tuples(v):
        return tuple(tuples(x) for x in v) if isinstance(v, list) else v
    for key in ('numerical_support', 'q_grouping', 'kpoints_cart', 'factor_shapes'):
        value[key] = tuples(value[key])
    return RSGDFFitSource(**value)


def _contract_jk(operator, densities, weights):
    """Rebuild retained-factor J/K in bounded auxiliary panels, no native calls.

    Uses the same canonical Gram convention as periodic_fitted_support. Retained
    factors, spin densities and the resulting Focks remain resident; panel
    staging is separate and is not a total process memory bound.
    """
    nk, n, _ = densities[0].shape
    total = densities[0] + densities[1]
    j = np.zeros_like(total, dtype=complex)
    ks = [np.zeros_like(total, dtype=complex), np.zeros_like(total, dtype=complex)]
    panel = max(1, 1024**2 // max(16*n*n, 1))
    rank = operator[0, 0].shape[0]
    for first in range(0, rank, panel):
        stop = min(first + panel, rank)
        rho = sum(weights[k] * np.einsum('Ppq,pq->P', operator[k,k][first:stop].conj(), total[k]) for k in range(nk))
        for k in range(nk): j[k] += np.einsum('P,Ppq->pq', rho, operator[k,k][first:stop])
    for k in range(nk):
        for q in range(nk):
            factors = operator[k, q]
            for first in range(0, len(factors), panel):
                block = factors[first:first+panel]
                for spin in range(2):
                    ks[spin][k] += weights[q]*np.einsum('Ppq,qs,Prs->pr', block, densities[spin][q], block.conj(), optimize=True)
    hermitian = lambda a: .5 * (a + a.conj().transpose(0, 2, 1))
    return hermitian(j), tuple(hermitian(k) for k in ks)





def _physical_request_identity(physical):
    """Recompute the driver's request digest from full archived inputs."""
    g = physical['geometry']
    geometry = (int(g['dimension']), np.asarray(g['lattice_bohr'], dtype=float),
                [(int(atom['Z']), tuple(atom['xyz_bohr'])) for atom in g['nuclei']],
                int(g['charge']), int(g['multiplicity']))
    return _digest((geometry, physical['basis']['sha256'], tuple(physical['mesh']),
                    physical['options'], physical['method'], physical['controls'],
                    'finite-character/full-Coulomb/BvK-ewald'))


def _physical_source_signature(physical, auxiliary_sha256):
    geometry = physical['geometry']
    digest = hashlib.sha256(b'vibeqc-rsgdf-source-v1\0')
    digest.update(str(int(geometry['dimension'])).encode('ascii'))
    digest.update(np.asarray(geometry['lattice_bohr'],dtype='<f8').tobytes())
    for atom in geometry['nuclei']:
        digest.update(str(int(atom['Z'])).encode('ascii')); digest.update(b':')
        digest.update(np.asarray(atom['xyz_bohr'],dtype='<f8').tobytes())
    digest.update(physical['basis']['sha256'].encode('ascii'))
    digest.update(auxiliary_sha256.encode('ascii'))
    return digest.hexdigest()


def _validate_input_record(physical, operator):
    """Check archived geometry, shells and complete characters without builders."""
    geometry = physical['geometry']; basis = physical['basis']; mesh = physical['mesh']
    labels = (basis['name'], physical['controls'].get('aux_basis'),
              operator.descriptor.orbital_basis, operator.descriptor.auxiliary_basis)
    if any(label is not None and (not isinstance(label, str) or '/' in label or '\\' in label
                                  or label.startswith('~') or ':' in label) for label in labels):
        raise ValueError('restart v1 requires portable registered basis labels, not filesystem paths')
    lattice = np.asarray(geometry['lattice_bohr'], dtype=float)
    if (lattice.shape != (3,3) or not np.isfinite(lattice).all()
            or abs(np.linalg.det(lattice)) < 1e-12
            or not isinstance(mesh, (list,tuple)) or len(mesh) != 3
            or any(type(n) is not int or n < 1 for n in mesh)):
        raise ValueError('invalid restart lattice/mesh')
    shells = []
    for record in basis['shells']:
        if set(record) != {'atom_index','l','pure','exponents','coefficients','origin'}:
            raise ValueError('invalid normalized restart shell')
        for key in ('exponents','coefficients','origin'):
            if not np.isfinite(record[key]).all(): raise ValueError('nonfinite restart basis')
        if len(record['origin']) != 3: raise ValueError('invalid restart shell origin')
        shells.append(SimpleNamespace(**record))
    rebuilt = _basis_record(SimpleNamespace(name=basis['name'], nbasis=basis['nbasis'], shells=lambda:shells))
    if rebuilt != basis: raise ValueError('restart normalized basis fingerprint mismatch')
    for atom in geometry['nuclei']:
        if (type(atom['Z']) is not int or atom['Z'] < 1 or len(atom['xyz_bohr']) != 3
                or not np.isfinite(atom['xyz_bohr']).all()):
            raise ValueError('invalid restart nuclear geometry')
    if operator.descriptor.source_signature != _physical_source_signature(physical, operator.descriptor.auxiliary_basis_sha256):
        raise ValueError('restart geometry/basis/auxiliary source signature mismatch')
    requested = physical['controls']
    executed = dict(operator.descriptor.numerical_support)
    descriptor = operator.descriptor
    if descriptor.schema != 'vibeqc.rsgdf.executed-source/v1' or descriptor.engine_version != 1:
        raise ValueError('unsupported restart source descriptor version')
    for scalar,field in (('metric_threshold','metric_threshold'),
                         ('base_reciprocal_cutoff_ha','reciprocal_cutoff_ha'),
                         ('executed_tail_reciprocal_cutoff_ha','tail_reciprocal_cutoff_ha')):
        if getattr(descriptor,scalar) != executed.get(field):
            raise ValueError('restart source descriptor scalar and executed support differ')
    if (requested.get('fit_source') != 'classic' or requested.get('backend') != 'ri'
            or requested.get('gdf_method') != 'rsgdf'
            or not isinstance(requested.get('coulomb_controls'), dict)
            or requested['coulomb_controls'] != executed):
        raise ValueError('restart requested Coulomb controls and executed source/support differ')
    for legacy, field in (('rsgdf_ke_cutoff','reciprocal_cutoff_ha'),
                          ('rsgdf_tail_ke_cutoff','tail_reciprocal_cutoff_ha'),
                          ('gdf_linear_dep_threshold','metric_threshold')):
        if requested.get(legacy) != executed.get(field):
            raise ValueError('restart normalized requested controls and source differ')
    if requested.get('support_policy') not in (None, executed.get('support_policy')):
        raise ValueError('restart requested support policy differs from executed source')
    domains = json.loads(operator.descriptor.executed_domains_json)
    if domains.get('schema') != 'vibeqc.chi.fitted-domains/v1':
        raise ValueError('unsupported restart fitted-domain schema; no implicit migration')
    if domains.get('controls') != executed:
        raise ValueError('restart factor support descriptor and executed domains differ')
    records = domains['q_records']
    if any(np.shape(record['lattice']) != (3,3) or not np.allclose(record['lattice'],lattice,atol=1e-13,rtol=0) for record in records):
        raise ValueError('restart source and physical lattice mismatch')
    fractional = np.asarray(operator.descriptor.kpoints_cart) @ lattice/(2*np.pi)
    residues = np.mod(np.rint(fractional*np.asarray(mesh)).astype(int),mesh)
    if (not np.allclose(fractional*np.asarray(mesh),np.rint(fractional*np.asarray(mesh)),atol=1e-10,rtol=0)
            or len(set(map(tuple,residues))) != int(np.prod(mesh))):
        raise ValueError('restart characters do not cover the declared mesh')


def _validate_overlap_space(hf, method, overlaps):
    """Retain the actual canonical AO space; never silently restore lost modes."""
    summary = getattr(hf, 'linear_dependence', None)
    channels = ('mo_coeffs',) if method != 'UHF' else ('mo_coeffs_alpha','mo_coeffs_beta')
    threshold = 1e-8 if method == 'ROHF' else (None if summary is None else float(summary.threshold))
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError('invalid restart overlap truncation threshold')
    ranks = []
    for k,s in enumerate(overlaps):
        if threshold is None:
            # Legacy records without a declared projection can only represent
            # the complete positive AO metric, never an undocumented truncation.
            if np.linalg.eigvalsh(s).min() <= 0:
                raise ValueError('restart reduced overlap needs its executed rank contract')
            x = np.eye(len(s)); rank = len(s)
        else:
            diagonal = np.diag(s).real
            scale = 1/np.sqrt(diagonal) if method != 'ROHF' and np.all(diagonal > 0) else np.ones(len(s))
            eigenvalues,eigenvectors = np.linalg.eigh(scale[:,None]*s*scale[None,:])
            keep = eigenvalues >= threshold
            rank = int(keep.sum()); x = scale[:,None]*eigenvectors[:,keep]
            if rank == 0: raise ValueError('restart overlap discarded every AO mode')
        ranks.append(rank)
        for name in channels:
            c = np.asarray(getattr(hf,name))[k]
            if c.shape != (len(s),rank): raise ValueError('restart orbital retained-rank mismatch')
            qc = np.linalg.qr(c)[0]; qx = np.linalg.qr(x)[0]
            if np.max(abs(qc@qc.conj().T-qx@qx.conj().T)) > 1e-9:
                raise ValueError('restart orbital retained-space projector mismatch')
    if summary is not None and (int(summary.n_basis) != len(overlaps[0]) or list(summary.n_kept_per_k) != ranks):
        raise ValueError('restart overlap rank metadata mismatch')


def _validate_physical(hf, method, physical, *, madelung=None):
    from .canonical_mp2 import CanonicalFactorAdapter, _spin_state
    op = hf.fitted_operator
    if (op.descriptor.source != 'classic' or op.kernel != 'coulomb'
            or getattr(hf, 'smearing_temperature', 0) != 0
            or getattr(hf, 'executed_chi_xc', None) is not None):
        raise NotImplementedError('restart supports classic unsmeared HF only')
    if (hf.aiccm2026dev_b.electronic_method != method or hf.aiccm2026dev_b.backend != 'ri'
            or hf.exact_exchange_assembly.c_full != 1 or hf.exact_exchange_assembly.c_sr != 0
            or getattr(hf, 'returned_operator', 'iterated') != 'iterated'
            or getattr(hf, 'energy_operator', 'iterated') != 'iterated'):
        raise ValueError('restart method/operator contract mismatch')
    from .ecp import validate_ecp_reference
    ecp = validate_ecp_reference(hf)
    if physical.get('ecp') != ('all-electron' if ecp is None else
            {k:v for k,v in ecp.items() if k not in ('executed_support', 'energy_components')}):
        raise ValueError('restart ECP physical identity mismatch')
    adapter = CanonicalFactorAdapter(op)
    source_record = hf.executed_fit_source.as_dict(); operator_record = op.descriptor.as_dict()
    source_record.pop('executed_domains_json',None); operator_record.pop('executed_domains_json',None)
    if source_record != operator_record:
        raise ValueError('restart SCF and factor source descriptors differ')
    if json.loads(hf.executed_fit_source.executed_domains_json).get('fitted_operator_identity') != op.identity:
        raise ValueError('restart executed source/operator identity mismatch')
    a = _state_arrays(hf, method)
    s, h, weights = a['overlap'], a['hcore'], a['kpoint_weights']
    nk = adapter.nk; n = op[0,0].shape[-1]
    if (s.shape != (nk,n,n) or h.shape != s.shape or weights.shape != (nk,)
            or np.max(abs(weights-np.asarray(op._weights))) > 1e-14
            or a['kpoints_cart'].shape != (nk,3)
            or np.max(abs(a['kpoints_cart']-op.descriptor.kpoints_cart)) > 1e-14):
        raise ValueError('restart character/metric/operator dimensions mismatch')
    if min(np.linalg.eigvalsh(s).min(), 0.) < -1e-10 or np.max(abs(s-s.conj().transpose(0,2,1))) > 1e-10:
        raise ValueError('restart overlap is not a Hermitian positive metric')
    _validate_overlap_space(hf, method, s)
    for name, value in a.items():
        if value.dtype.kind not in 'biufc' or not np.isfinite(value).all():
            raise ValueError('nonfinite or nonnumeric restart physical state')
    densities = (a['density']/2, a['density']/2) if method == 'RHF' else (a['density_alpha'], a['density_beta'])
    focks = (a['fock'], a['fock']) if method == 'RHF' else (a['fock_alpha'], a['fock_beta'])
    j, k = _contract_jk(op, densities, weights)
    seams = tuple(s @ d @ s for d in densities)
    base = tuple(h + j - x for x in k)
    if madelung is None:
        # Preserve the actually executed scalar q=0 seam, then independently
        # check every matrix element against it. No default seam is substituted.
        norm = sum(np.vdot(x, x).real for x in seams)
        if norm <= 0: raise ValueError('restart has no occupied projector')
        madelung = float(sum(np.vdot(x, b-f).real for x,b,f in zip(seams,base,focks))/norm)
    rebuilt = tuple(b-madelung*x for b,x in zip(base,seams))
    fock_error = max(float(np.max(abs(f-r))) for f,r in zip(focks,rebuilt))
    energy = float(hf.e_nuclear) + sum(float(np.einsum('k,kpq,kqp->', weights,d,h+f).real)*.5 for d,f in zip(densities,rebuilt))
    if fock_error > 1e-9 or abs(energy-float(hf.energy)) > 1e-11:
        raise ValueError('restart fails retained-source physical Fock/energy rebuild')
    _validate_input_record(physical, op)
    g = physical['geometry']; nbasis = physical['basis']['nbasis']
    explicit = g['n_electrons'] if ecp is None else ecp['explicit_electrons']
    na = (explicit+g['multiplicity']-1)//2; nb = explicit-na
    if (g['dimension'] != 3 or g['charge'] != 0
            or nbasis != n or g['n_electrons'] != sum(x['Z'] for x in g['nuclei'])-g['charge']
            or (explicit+g['multiplicity']-1)%2 or not 0 <= nb <= na <= n):
        raise ValueError('restart geometry/electron/spin/basis contract mismatch')
    if (op.descriptor.orbital_basis_sha256 != physical['basis']['sha256']
            or physical['method'] != method or np.prod(physical['mesh']) != nk):
        raise ValueError('restart basis/source/spin/mesh identity mismatch')
    actual = [float(np.einsum('k,kpq,kqp->', weights, d, s).real) for d in densities]
    if max(abs(actual[0]-na), abs(actual[1]-nb)) > 1e-9:
        raise ValueError('restart electron/spin count mismatch')
    if method == 'ROHF':
        from .rohf import validate_rohf_state
        residuals = validate_rohf_state(hf)
        orbital = max(residuals.values())
        from ...rohf import roothaan_effective_fock
        effective = np.asarray([roothaan_effective_fock(fa,fb,da,db,sk)
            for fa,fb,da,db,sk in zip(*rebuilt,*densities,s)])
        c = np.asarray(hf.mo_coeffs); eps = np.asarray(hf.mo_energies)
        if (np.max(abs(effective-np.asarray(hf.fock))) > 1e-9
                or np.max(abs(effective@c-(s@c)*eps[:,None,:])) > 1e-9):
            raise ValueError('ROHF restart effective Fock/canonicalization mismatch')
        if hf.n_alpha != na or hf.n_beta != nb: raise ValueError('ROHF restart spin mismatch')
    else:
        channels = _spin_state(hf, method, 1e-9)
        if tuple(c[2] for c in channels) != (na,nb): raise ValueError('restart occupation/spin mismatch')
        orbital = max(float(np.max(abs(f@d@s-s@d@f))) for f,d in zip(rebuilt,densities))
        if orbital > 1e-9: raise ValueError('restart projector is not stationary')
    return dict(madelung=float(madelung), physical_energy=energy, fock_error=fock_error,
                orbital_residual=float(orbital),
                generalized_rohf_gradient_norm=float(orbital) if method == 'ROHF' else 0.)


def save_chi_hf_reference(path, reference, *, system=None, basis=None):
    """Persist an accepted classic state and its exact retained source."""
    if not isinstance(reference, ChiHFReference): raise TypeError('expected ChiHFReference')
    hf = reference.validate(reference.request_identity, reference.method)
    physical = getattr(hf, '_chi_restart_physical', None)
    if physical is None:
        raise ValueError('restart requires bind_chi_restart_inputs with original full request controls')
    if _physical_request_identity(physical) != reference.request_identity:
        raise ValueError('restart request identity does not match archived physical inputs')
    if system is not None and physical['geometry'] != _geometry_record(system):
        raise ValueError('restart geometry mismatch')
    if basis is not None and physical['basis'] != _basis_record(basis):
        raise ValueError('restart basis mismatch')
    checked = _validate_physical(hf, reference.method, physical)
    arrays = {}
    state = dict(_state_arrays(hf, reference.method))
    state.update({k: getattr(hf,k) for k in _VALUE_FIELDS if hasattr(hf,k)})
    # Persist immutable factors separately, in deterministic character order.
    for (i,j), factor in sorted(hf.fitted_operator.items()): arrays[f'factor_{i}_{j}'] = factor
    schema = _SCHEMA if physical.get('ecp') == 'all-electron' else _ECP_SCHEMA
    metadata = dict(schema=schema, kind='accepted-reference', method=reference.method,
        request_identity=reference.request_identity, physical_identity=portable_reference_identity(reference),
        physical=_encode(physical, arrays), state=_encode(state, arrays),
        operator=hf.fitted_operator.descriptor.as_dict(), source=hf.executed_fit_source.as_dict(),
        madelung=checked['madelung'], integrity_is_acceptance=False,
        quantitative_status='not-qualified')
    return write_restart_state(path, metadata, arrays)


def load_chi_hf_reference(path, *, expected_request_identity=None, expected_method=None,
                          system=None, basis=None, mode='accepted'):
    """Validate/reseal an accepted reference or explicitly return only a guess.

    ``system``/``basis`` validate the supplied physical input. Exact numerical
    controls are checked with ``expected_request_identity`` or by the unchanged
    driver ``hf_reference`` validation before reuse. No version migration, SCF,
    native integration, source refit, rank change or automatic downgrade occurs.
    """
    if mode not in ('accepted', 'guess'): raise ValueError("restart mode must be 'accepted' or 'guess'")
    metadata, arrays = read_restart_state(path)
    required = {'schema','kind','method','request_identity','physical_identity','physical','state',
                'operator','source','madelung','integrity_is_acceptance','quantitative_status'}
    if set(metadata) != required:
        raise ValueError('chi HF restart metadata fields mismatch')
    if metadata.get('schema') not in (_SCHEMA, _ECP_SCHEMA) or metadata.get('kind') != 'accepted-reference':
        raise ValueError('unsupported chi HF restart schema or state kind')
    method = metadata.get('method')
    if method not in ('RHF','UHF','ROHF'): raise ValueError('unsupported restart HF method')
    if expected_method is not None and method != str(expected_method).upper(): raise ValueError('restart spin/method mismatch')
    if expected_request_identity is not None and metadata['request_identity'] != expected_request_identity:
        raise ValueError('restart request controls/source compatibility mismatch')
    physical = _decode(metadata['physical'], arrays)
    if (metadata['schema'] == _SCHEMA) != (physical.get('ecp') == 'all-electron'):
        raise ValueError('restart semantic version does not match its ECP Hamiltonian')
    if _physical_request_identity(physical) != metadata['request_identity']:
        raise ValueError('restart request identity does not match archived physical inputs')
    if system is not None and physical['geometry'] != _geometry_record(system): raise ValueError('restart geometry/spin mismatch')
    if basis is not None and physical['basis'] != _basis_record(basis): raise ValueError('restart basis mismatch')
    descriptor = _source(metadata['operator']); source = _source(metadata['source'])
    factors = {}
    for pair, shape in descriptor.factor_shapes:
        name = f'factor_{pair[0]}_{pair[1]}'
        if name not in arrays or arrays[name].shape != shape: raise ValueError('restart factors missing or shape mismatch')
        factors[tuple(pair)] = arrays[name]
    def array_references(value):
        if isinstance(value, dict):
            if set(value) == {'array'}: return {value['array']}
            return set().union(*(array_references(v) for v in value.values()))
        if isinstance(value, list): return set().union(*(array_references(v) for v in value))
        return set()
    used_arrays = {f'factor_{i}_{j}' for i,j in factors}
    used_arrays |= array_references(metadata['state']) | array_references(metadata['physical'])
    if used_arrays != set(arrays): raise ValueError('chi HF restart numeric coverage mismatch')
    state = _decode(metadata['state'], arrays)
    hf = SimpleNamespace(**state)
    hf.fitted_operator = ChiFittedOperator(factors, descriptor, hf.kpoint_weights)
    hf.executed_fit_source = source
    hf.converged = True
    hf._chi_hf_request_identity = metadata['request_identity']
    hf._chi_restart_physical = physical
    hf.process_provenance = {}
    temporary = ChiHFReference(hf, 0, metadata['request_identity'], '', method)
    if portable_reference_identity(temporary) != metadata['physical_identity']:
        raise ValueError('restart portable physical identity mismatch')
    checked = _validate_physical(hf, method, physical, madelung=metadata['madelung'])
    if mode == 'guess':
        density = hf.density if method == 'RHF' else (hf.density_alpha, hf.density_beta)
        coeff = hf.mo_coeffs if method != 'UHF' else (hf.mo_coeffs_alpha, hf.mo_coeffs_beta)
        return ChiHFGuess(method, density, coeff, metadata['physical_identity'])
    # Measured current checks replace serialized acceptance flags. Historical
    # energy change is not claimed as a new SCF iteration convergence result.
    hf.chi_scf_acceptance = dict(final_rebuild=True, stationary=True,
        energy_residual=0., energy_consistency_error=abs(checked['physical_energy']-hf.energy),
        orbital_residual=checked['orbital_residual'], density_refill_residual=0.,
        generalized_rohf_gradient_norm=checked['generalized_rohf_gradient_norm'],
        orbital_tolerance=1e-9, energy_tolerance=1e-11, stability='not-assessed',
        restart_validation='retained-source Fock/energy and occupied-projector rebuild',
        convergence_origin='reconstructed accepted stationary reference; no SCF iterations')
    # Public output adapters select Gamma from per-character lists. Preserve
    # that established presentation without changing the normalized identity.
    for name in _ARRAY_FIELDS:
        if name not in ('kpoints_cart','kpoint_weights') and hasattr(hf,name):
            setattr(hf,name,[row for row in np.asarray(getattr(hf,name))])
    hf.accepted_density_k = hf.density if method == 'RHF' else (hf.density_alpha,hf.density_beta)
    hf.scf_trace = []
    hf.n_iter = int(getattr(hf, 'n_iter', 0))
    hf.restart = dict(mode='accepted', status='restored-validated-reference',
        new_scf_iterations=0, original_scf_iterations=hf.n_iter,
        physical_identity=metadata['physical_identity'], source_identity=descriptor.identity)
    from . import reference as refs
    hf.process_provenance = refs.loaded_process_provenance()
    return refs._seal_chi_hf_result(hf, method)


__all__ = ['ChiHFGuess', 'bind_chi_restart_inputs', 'portable_reference_identity',
           'save_chi_hf_reference', 'load_chi_hf_reference']
