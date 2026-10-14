"""Inert scalar-semilocal ECP identities for the experimental classic chi route.

The Hamiltonian is T + V_ion(Z_eff) + U_local + sum_lm |lm> U_l <lm|.
The inline libecpint convention stores the local channel at the largest l;
its zero-coefficient placeholder is physical metadata and must not be dropped.
This module validates metadata and retained arrays; it never builds integrals.
"""
from collections.abc import Mapping
import hashlib
import json
import numbers
from types import SimpleNamespace

import numpy as np

_SCHEMA = 'vibeqc.chi.scalar-ecp/v1'
_COMPONENTS = ('kinetic', 'ionic_coulomb', 'ecp_local', 'ecp_nonlocal')
_EXECUTION_KEYS = frozenset(('executed_support', 'energy_components'))


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f'chi ECP {name} must be an integer >= {minimum}')
    return int(value)


def _coordinates(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f'chi ECP {name} needs three finite coordinates')
    return a.tolist()


def primitive_record(block):
    """Serialize exactly the native primitive arrays, including zero terms."""
    count = _integer(_get(block, 'n_primitive'), 'primitive count', 1)
    result = dict(n_primitive=count)
    for field in ('exponents', 'coefficients', 'ams', 'ns'):
        values = list(_get(block, field, ()) or ())
        if len(values) != count:
            raise ValueError('chi ECP primitive array lengths disagree')
        if field in ('ams', 'ns'):
            result[field] = [_integer(v, field) for v in values]
        else:
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not np.isfinite(v) for v in values):
                raise ValueError('chi ECP primitives must be finite real numbers')
            result[field] = list(map(float, values))
    if min(result['exponents']) <= 0:
        raise ValueError('chi ECP primitive exponents must be positive')
    return result


def _identity(record):
    base = {k: v for k, v in record.items() if k not in _EXECUTION_KEYS and k != 'identity'}
    return hashlib.sha256(json.dumps(base, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ecp_contract(system, options):
    """Validate one complete inline scalar ECP request, or return None for AE.

    Bare nuclear labels remain intact. ECP cores are subtracted exactly once
    from their physical electron count; later frozen cores concern only the
    remaining explicit occupied orbitals. XML recipes and spin-orbit data are
    not this periodic inline route.
    """
    for name in ('ecp_spin_orbit', 'spin_orbit_ecp', 'ecp_so', 'ecp_so_blocks',
                 'ecp_so_primitive_blocks', 'spin_orbit', 'spin_orbit_coupling'):
        if _get(options, name, False):
            raise NotImplementedError('chi spin-orbit ECPs are unsupported')
    for name in ('ecp_centers', 'ecp_library', 'ecp_primitive_centers'):
        if _get(options, name, None):
            raise NotImplementedError('chi ECP requires inline periodic primitives and home centers')
    blocks = list(_get(options, 'ecp_primitive_blocks', ()) or ())
    centers = list(_get(options, 'ecp_home_centers', ()) or ())
    effective = list(_get(options, 'ecp_effective_charges', ()) or ())
    core_value = _get(options, 'ecp_total_ncore', 0)
    absent_core = core_value is None or (not isinstance(core_value, (bool, np.bool_))
        and isinstance(core_value, numbers.Real) and core_value == 0)
    if not blocks and not centers and not effective and absent_core:
        return None
    atoms = list(system.unit_cell)
    bare = [_integer(a.Z, 'bare nuclear charge', 1) for a in atoms]
    xyz = [_coordinates(a.xyz, 'nuclear center') for a in atoms]
    charge = _integer(abs(system.charge), 'absolute cell charge') * (-1 if system.charge < 0 else 1)
    multiplicity = _integer(system.multiplicity, 'multiplicity', 1)
    physical = _integer(system.n_electrons(), 'physical electron count')
    if physical != sum(bare)-charge:
        raise ValueError('chi ECP physical electron count disagrees with bare nuclei and charge')
    core = _integer(0 if core_value is None else core_value, 'removed core count')
    if not blocks:
        if centers or core or (effective and effective != bare):
            raise ValueError('chi ECP metadata requires a complete explicit operator')
        return None
    if int(system.dim) != 3 or charge != 0:
        raise NotImplementedError('chi scalar ECP requires neutral three-dimensional cells')
    if len(blocks) != len(centers) or len(effective) != len(atoms):
        raise ValueError('chi ECP blocks/centers/charges have inconsistent lengths')
    effective = [_integer(q, 'effective ionic charge', 1) for q in effective]
    if any(q > z for q,z in zip(effective,bare)):
        raise ValueError('chi ECP effective charge exceeds bare nuclear charge')
    removed = [z-q for z,q in zip(bare,effective)]
    if sum(removed) != core or core > physical:
        raise ValueError('chi ECP removed-core count disagrees with effective charges')
    projectors = []
    mapped = set()
    for block, center in zip(blocks,centers):
        center = _coordinates(center, 'home center')
        matches = [i for i, position in enumerate(xyz) if np.max(abs(np.asarray(center)-position)) <= 1e-10]
        if len(matches) != 1 or matches[0] in mapped:
            raise ValueError('chi ECP home center must map uniquely to one primitive-cell atom')
        atom = matches[0]; mapped.add(atom)
        b = primitive_record(block)
        projectors.append(dict(atom_index=atom, center_bohr=center,
            primitives=[[n,l,a,c] for n,l,a,c in zip(b['ns'],b['ams'],b['exponents'],b['coefficients'])],
            local_l=max(b['ams'])))
    if any(n and i not in mapped for i,n in enumerate(removed)):
        raise ValueError('chi ECP removed core has no matching operator')
    explicit = physical-core
    if explicit < multiplicity-1 or (explicit+multiplicity-1) % 2:
        raise ValueError('chi ECP explicit electron count and spin are inconsistent')
    record = dict(schema=_SCHEMA, operator='scalar-semilocal',
        atoms=[dict(Z=z,xyz_bohr=x,effective_charge=q,core_electrons=n)
               for z,x,q,n in zip(bare,xyz,effective,removed)],
        projectors=projectors, physical_electrons=physical, explicit_electrons=explicit,
        ecp_core_electrons=core, charge=charge, multiplicity=multiplicity)
    record['identity'] = _identity(record)
    return record


def _from_physical(physical):
    geometry = physical['geometry']
    system = SimpleNamespace(dim=geometry['dimension'], charge=geometry['charge'],
        multiplicity=geometry['multiplicity'],
        unit_cell=[SimpleNamespace(Z=a['Z'], xyz=a['xyz_bohr']) for a in geometry['nuclei']],
        n_electrons=lambda: geometry['n_electrons'])
    return ecp_contract(system, physical.get('options'))


def validate_ecp_reference(hf):
    """Check actual ECP construction, populations and retained component arrays.

    Storage integrity and this internal agreement are not scientific acceptance.
    An ECP keyword or nonzero core count without executed construction refuses.
    """
    record = getattr(hf, 'chi_ecp', None)
    diagnostics = getattr(hf, 'aiccm2026dev_b', None)
    core = getattr(hf, 'ecp_total_ncore', 0)
    diagnostic_core = _get(diagnostics, 'ecp_total_ncore', 0)
    physical = getattr(hf, '_chi_restart_physical', None)
    requested = _from_physical(physical) if physical is not None else None
    if record is None:
        if core or diagnostic_core or requested is not None:
            raise ValueError('chi ECP reference lacks complete executed operator metadata')
        return None
    if not isinstance(record, Mapping) or record.get('schema') != _SCHEMA or record.get('identity') != _identity(record):
        raise ValueError('chi ECP identity is invalid')
    base = {k:v for k,v in record.items() if k not in _EXECUTION_KEYS}
    if requested is None or base != requested:
        raise ValueError('chi ECP executed operator differs from original full request')
    if (core != record['ecp_core_electrons'] or diagnostic_core != core
            or getattr(hf, 'effective_n_electrons', None) != record['explicit_electrons']
            or list(getattr(hf, 'ecp_effective_charges', ())) != [a['effective_charge'] for a in record['atoms']]):
        raise ValueError('chi ECP effective electron/charge bookkeeping disagrees')
    expected_charges = [a['effective_charge'] for a in record['atoms']]
    for owner, checks in ((hf, dict(effective_nuclear_charges=expected_charges,
            effective_nuclear_charge=sum(expected_charges))), (diagnostics, dict(
            physical_electron_count=record['physical_electrons'],
            effective_electron_count=record['explicit_electrons'],
            effective_nuclear_charges=expected_charges,
            effective_nuclear_charge=sum(expected_charges), effective_net_charge=record['charge']))):
        for key, expected in checks.items():
            observed = _get(owner, key)
            if observed is not None and (list(observed) if isinstance(expected,list) else observed) != expected:
                raise ValueError('chi ECP diagnostic charge/electron bookkeeping disagrees')
    source = getattr(hf, 'executed_fit_source', None)
    operator = getattr(hf, 'fitted_operator', None)
    if (getattr(source, 'source', None) != 'classic'
            or getattr(getattr(operator, 'descriptor', None), 'source', None) != 'classic'):
        raise NotImplementedError('chi ECP references require classic retained Coulomb factors; SR is unsupported')
    domains = json.loads(source.executed_domains_json)
    one_electron = domains.get('one_electron')
    recorded = one_electron.get('ecp') if isinstance(one_electron, Mapping) else None
    source_record = {k:v for k,v in record.items() if k != 'energy_components'}
    if not isinstance(record.get('executed_support'), Mapping) or recorded != source_record:
        raise ValueError('chi ECP executed support/source identity disagrees')
    components = getattr(hf, 'hcore_components', None)
    if not isinstance(components, Mapping) or set(components) != set(_COMPONENTS):
        raise ValueError('chi ECP requires separate retained one-electron components')
    h = np.asarray(hf.hcore); s = np.asarray(hf.overlap); weights = np.asarray(hf.kpoint_weights)
    if (h.ndim != 3 or h.shape[1] != h.shape[2] or s.shape != h.shape
            or weights.shape != (len(h),) or not np.isfinite(h).all()
            or not np.isfinite(s).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or abs(float(weights.sum())-1) > 1e-12
            or np.max(abs(h-h.conj().transpose(0,2,1))) > 1e-10
            or np.max(abs(s-s.conj().transpose(0,2,1))) > 1e-10):
        raise ValueError('chi ECP Hamiltonian/overlap/character weights are invalid')
    arrays = [np.asarray(components[k]) for k in _COMPONENTS]
    if any(a.shape != h.shape or not np.isfinite(a).all() or np.max(abs(a-a.conj().transpose(0,2,1))) > 1e-10 for a in arrays):
        raise ValueError('chi ECP component dimensions or Hermiticity invalid')
    if np.max(abs(sum(arrays)-h)) > 1e-10:
        raise ValueError('chi ECP hcore components do not sum once to the executed Hamiltonian')
    method = _get(diagnostics, 'electronic_method')
    spin = record['multiplicity']-1
    na = (record['explicit_electrons']+spin)//2; nb = record['explicit_electrons']-na
    if method == 'RHF':
        if na != nb:
            raise ValueError('chi ECP RHF requires equal spin populations')
        densities = [np.asarray(hf.density)/2]*2
        occupations = [np.asarray(hf.occupations)/2]*2
    else:
        densities = [np.asarray(getattr(hf,'density_'+spin)) for spin in ('alpha','beta')]
        occupations = [np.asarray(getattr(hf,'occupations_'+spin)) for spin in ('alpha','beta')]
    for d,occ,count in zip(densities,occupations,(na,nb)):
        if (d.shape != h.shape or weights.shape != (len(h),) or occ.shape[0] != len(h)
                or not np.isfinite(d).all() or not np.isfinite(occ).all()
                or np.max(abs(occ-np.rint(occ))) > 1e-12 or np.any(occ < 0) or np.any(occ > 1)
                or np.max(abs(occ.sum(axis=1)-count)) > 1e-10
                or np.max(abs(np.einsum('kpq,kqp->k',d,s)-count)) > 1e-9):
            raise ValueError('chi ECP explicit populations disagree with occupied state')
    total = sum(densities)
    energies = record.get('energy_components')
    if not isinstance(energies, Mapping) or set(energies) != set(_COMPONENTS) | {'ionic_ewald'}:
        raise ValueError('chi ECP energy components are incomplete')
    computed = {key: float(np.einsum('k,kpq,kqp->',weights,total,a).real) for key,a in zip(_COMPONENTS,arrays)}
    computed['ionic_ewald'] = float(hf.e_nuclear)
    if any(not isinstance(energies[k], numbers.Real) or not np.isfinite(energies[k]) or abs(energies[k]-value) > 1e-10 for k,value in computed.items()):
        raise ValueError('chi ECP component energies disagree with the accepted density')
    return record


def electron_accounting(hf, frozen, occupied):
    """Report ECP-replaced and explicitly frozen electrons as distinct counts."""
    record = validate_ecp_reference(hf)
    explicit = sum(map(int,occupied))
    if any(f < 0 or f > n for f,n in zip(frozen,occupied)):
        raise ValueError('frozen core removes more than the explicit occupied electrons')
    removed = record['ecp_core_electrons'] if record else 0
    return dict(physical_electrons=explicit+removed, ecp_core_electrons=removed,
        explicit_electrons=explicit, frozen_explicit_electrons=sum(map(int,frozen)),
        correlated_electrons=explicit-sum(map(int,frozen)),
        occupied_per_spin=list(map(int,occupied)), frozen_explicit_per_spin=list(map(int,frozen)),
        correlated_per_spin=[int(n-f) for n,f in zip(occupied,frozen)],
        frozen_convention='explicit occupied orbitals only; ECP cores already removed')
