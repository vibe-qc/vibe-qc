"""Scalar semilocal ECP assembly for the explicit classic chi Hamiltonian.

Shaw and Hill, JCP 147, 074108 (2017), doi:10.1063/1.4986887, Eqs. 1-3:
the effective-charge Coulomb potential and the short-range local/nonlocal
Gaussian terms are distinct operators. Neither includes the other twice.
"""
from copy import deepcopy
import json

import numpy as np


def require_ecp_support(system, options, controls, source, *, basis=None):
    """Validate the operator before any native allocation or source reuse."""
    from .ecp import ecp_contract
    record = ecp_contract(system, options)
    if record is None:
        if basis is not None:
            from ...basis_registry import basis_replaces_core
            if basis_replaces_core(str(basis.name), {int(a.Z) for a in system.unit_cell}):
                raise ValueError('chi ECP-paired basis requires complete scalar ECP primitive options')
        return None
    if (int(system.dim) != 3 or int(system.charge) != 0 or source != 'classic'
            or controls is None or controls.support_policy != 'physical_pairs'):
        raise NotImplementedError('chi scalar ECP requires neutral 3D classic physical-pair fitted HF; SR ECP is unsupported')
    return record


def build_ecp_components(basis, system, options, context, contract):
    """Evaluate local and angular-projector terms through the native kernel.

The largest angular channel is local. Keep its zero placeholder even for a
pure nonlocal component: removing it would relabel the highest projector as
local (CRYSTAL23 section 3.2, equations 3.16-3.19).
"""
    from ... import _vibeqc_core as core
    blocks, centers, _, _, _ = context
    components = {}
    for component in ('ecp_local', 'ecp_nonlocal'):
        selected = []
        for original in blocks:
            block = core.ECPPrimitiveBlock()
            block.n_primitive = int(original.n_primitive)
            block.exponents = list(original.exponents)
            block.ams = list(original.ams)
            block.ns = list(original.ns)
            local_l = max(block.ams)
            block.coefficients = [float(c) if ((int(l) == local_l) == (component == 'ecp_local')) else 0.
                                  for l, c in zip(original.ams, original.coefficients)]
            selected.append(block)
        components[component] = core.compute_ecp_lattice_from_primitives(
            basis, system, options, centers, selected, physical_pair_support=True)
    record = deepcopy(contract)
    record['executed_support'] = dict(
        schema='vibeqc.chi.scalar-ecp-support/v1',
        pair_selector='physical shell-center distance; pair-complete cell enclosure',
        projector_selector='PairNuclearImageSelector AO midpoint',
        pair_cutoff_bohr=float(options.cutoff_bohr),
        projector_cutoff_bohr=float(options.nuclear_cutoff_bohr),
        lattice_bohr=np.asarray(system.lattice, dtype=float).tolist(),
        shell_origins_bohr=[list(s.origin) for s in basis.shells()],
        cells=[dict(index=list(map(int, c.index)), cart_bohr=list(c.r_cart))
               for c in components['ecp_local'].cells],
        membership='lossless native selector specification before screening, not integral-call trace',
        screening='native libecpint exact-zero shell-pair predicate',
        phase='exp(+i k dot translated ket cell)',
        normalization='native normalized spherical AO; energy per primitive cell')
    return components, record


def bloch_ecp_components(operators, kpoints):
    """Fold the actual finite operator; refuse a broken Hermitian domain."""
    from ..._vibeqc_core import bloch_sum
    result = {}
    for name, operator in operators.items():
        values = np.asarray([np.asarray(bloch_sum(operator, k)) for k in kpoints])
        if not np.isfinite(values).all():
            raise ValueError('nonfinite chi ECP one-electron operator')
        error = np.max(abs(values-values.conj().swapaxes(-1, -2)), initial=0.)
        if error > 1e-9:
            raise ValueError(f'chi ECP {name} violates physical Hermiticity: {error:g}')
        result[name] = .5*(values+values.conj().swapaxes(-1, -2))
    return result


def attach_ecp_result(result, record, components):
    """Record the executed scalar Hamiltonian and density-derived components."""
    if record is None:
        return result
    weights = np.asarray(result.kpoint_weights)
    density = (np.asarray(result.density_alpha)+np.asarray(result.density_beta)
               if hasattr(result, 'density_alpha') else np.asarray(result.density))
    energy = {name: float(np.einsum('k,kpq,kqp->', weights, density, value).real)
              for name, value in components.items()}
    energy['ionic_ewald'] = float(result.e_nuclear)
    result.chi_ecp = deepcopy(record)
    result.chi_ecp['energy_components'] = energy
    result.hcore_components = {name: np.array(value, copy=True) for name, value in components.items()}
    result.ecp_total_ncore = int(record['ecp_core_electrons'])
    result.ecp_effective_charges = tuple(a['effective_charge'] for a in record['atoms'])
    result.effective_n_electrons = int(record['explicit_electrons'])
    return result


def record_ecp_support(one_electron_record, ecp_record, controls):
    record = json.loads(one_electron_record)
    record['ecp'] = ecp_record
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(encoded.encode()) > controls.support_record_byte_cap:
        raise MemoryError('complete chi ECP one-electron record exceeds support record budget')
    return encoded
