"""Experimental retained classic chi source for the shared translation MP2 engine.

The unitary finite Fourier transform is Nejad et al., JCP 163, 214107
(2025), Sec. II, Eqs. (2)-(6). The engine retains its Riplinger/Neese
(2013) PNO density and Nejad Sec. V occupied-Fock-coupled residual.
No SCF, integral build, refit, Fock shift or dense AO matrix fold occurs here.
"""
from dataclasses import dataclass
from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace

import numpy as np

from .canonical_mp2 import prepare_canonical_reference
from .correlation import require_public_mp2_controls, require_public_mp2_mesh
from .reference import ChiHFReference


_REAL_TOL = 1e-10


def _max(array):
    return float(np.max(np.abs(array), initial=0.))


def _block_operator(values, phase, mesh, label):
    """A[0,D] = N^-1 sum_k exp(-ikD) A(k), with a full inverse audit."""
    from ...dlpno.translation_operators import CyclicAOOperator, validate_translation_matrix
    blocks = np.einsum('kd,kab->dab', phase.conj(), values) / len(phase)
    imaginary = _max(blocks.imag)
    if imaginary > _REAL_TOL:
        raise ValueError(f'{label} has no admitted real finite-torus operator')
    operator = CyclicAOOperator(blocks.real, mesh)
    validate_translation_matrix(operator, mesh, _REAL_TOL,
                                positive=label == 'overlap', name=label)
    error = _max(np.einsum('kd,dab->kab', phase, operator.blocks) - values)
    if error > _REAL_TOL:
        raise ValueError(f'{label} Fourier round trip changes the reference operator')
    return operator, dict(imaginary=imaginary, round_trip=error)


def _real_occupied(coefficients, overlap_k, phase, label):
    """Realify a conjugation-invariant occupied subspace, never its columns.

    Z[R mu,k i] = exp(ikR) C[k mu,i]/sqrt(N) is S-orthonormal. If its
    projector is real, [Re Z, Im Z] spans the same real space. Its metric
    Gram has exactly rank(Z) unit eigenvalues and the remaining ones zero.
    We verify that property and the complex metric projector row by row.
    Frozen and active spaces are transformed and audited independently.
    """
    nk, nao, bands = coefficients.shape
    if bands == 0:
        return np.empty((nk*nao, 0)), dict(projector_error=0., orthogonality=0.)
    z = np.einsum('kr,kai->raki', phase, coefficients).reshape(nk*nao, nk*bands) / np.sqrt(nk)
    sz = np.einsum('kr,kai->raki', phase, overlap_k @ coefficients).reshape(z.shape) / np.sqrt(nk)
    real_span = np.concatenate((z.real, z.imag), axis=1)
    metric_span = np.concatenate((sz.real, sz.imag), axis=1)
    gram = real_span.T @ metric_span
    values, vectors = np.linalg.eigh(.5*(gram+gram.T))
    rank = z.shape[1]
    if (_max(values[:-rank]) > _REAL_TOL or _max(values[-rank:] - 1.) > _REAL_TOL):
        raise ValueError(f'{label} occupied projector does not admit a real frame')
    transform = vectors[:, -rank:] / np.sqrt(values[-rank:])
    frame = real_span @ transform
    metric_frame = metric_span @ transform
    projector_right = sz.conj().T
    error = 0.
    for start in range(0, len(z), nao):
        # The complex difference checks both Re and Im, without a dense AO projector.
        error = max(error, _max(frame[start:start+nao] @ metric_frame.T
                               - z[start:start+nao] @ projector_right))
    orthogonality = _max(frame.T @ metric_frame - np.eye(rank))
    if max(error, orthogonality) > _REAL_TOL:
        raise ValueError(f'{label} real frame changes the occupied metric projector')
    return frame, dict(projector_error=error, orthogonality=orthogonality)


def _home_factors(adapter, phase, cells, mesh):
    """Reuse the compact native transform and verify every input factor block.

    A common canonical primitive auxiliary frame is required. Reality and
    symmetry alone are insufficient: the forward transform must recover all
    retained L(k_i,k_a), including complex phases and discarded metric modes.
    A q-dependent gauge incompatible with this real frame fails closed.
    """
    from ...dlpno.translation_factors import TranslationHomeFactors
    from ..._vibeqc_core import aiccm2026dev_b_lpq_to_real_home_auxiliary
    home, imaginary, symmetry = aiccm2026dev_b_lpq_to_real_home_auxiliary(
        dict(adapter.operator), np.ascontiguousarray(adapter.k_fractional),
        np.asarray(cells, dtype=np.int64), np.asarray(mesh, dtype=np.int64))
    if not np.isfinite([imaginary, symmetry]).all() or max(imaginary, symmetry) > _REAL_TOL:
        raise ValueError('retained Coulomb factors have no admitted real symmetric auxiliary frame')
    nk, nao = len(phase), home.shape[1] // len(phase)
    grid = home.reshape(home.shape[0], nk, nao, nk, nao)
    error = 0.
    for i in range(nk):
        # One auxiliary/home-AO/translated-AO rectangle, not translated factors.
        left = np.einsum('r,Prasb->Pasb', phase[i].conj(), grid)
        for a in range(nk):
            restored = np.einsum('s,Pasb->Pab', phase[a], left)
            error = max(error, _max(restored - adapter.operator[i, a]))
    if error > _REAL_TOL:
        raise ValueError('Coulomb factor Fourier round trip changes the retained source')
    return TranslationHomeFactors(home, mesh), dict(
        imaginary=float(imaginary), symmetry=float(symmetry), round_trip=error)


def _layout(physical, n_cells):
    """Repeat sealed atom/shell metadata, without rebuilding a basis or integrals."""
    physical = deepcopy(physical)
    geometry, basis = physical['geometry'], physical['basis']
    lattice = np.asarray(geometry['lattice_bohr'])
    atoms, shells = [], []
    for cell in product(*(range(n) for n in physical['mesh'])):
        shift = lattice @ cell
        offset = len(atoms)
        atoms.extend(SimpleNamespace(Z=a['Z'], xyz=np.asarray(a['xyz_bohr'])+shift)
                     for a in geometry['nuclei'])
        shells.extend(SimpleNamespace(**{**s, 'atom_index': s['atom_index']+offset,
                                        'origin': np.asarray(s['origin'])+shift})
                      for s in basis['shells'])
    molecule = SimpleNamespace(atoms=tuple(atoms), charge=geometry['charge']*n_cells,
        multiplicity=1, n_electrons=lambda: geometry['n_electrons']*n_cells)
    repeated = SimpleNamespace(name=basis['name'], nbasis=basis['nbasis']*n_cells,
                               shells=lambda: tuple(shells))
    return molecule, repeated


def build_chi_translation_source(reference, *, frozen_core=0, factor_provider='streamed',
                                 max_factor_bytes=64*1024**2, factor_auxiliary_tile=16,
                                 factor_cache_bytes=1024**2, compact_occupied=False,
                                 max_selection_bytes=64*1024**2):
    """Adapt a sealed classic physical-pair RI RHF reference, with no SCF work.

    ``frozen_core`` counts lowest explicit occupied orbitals per character;
    the returned source puts the entire frozen projector before active ones.
    Only references with a verified real finite-torus representation enter.
    The sealed physical input binding supplies mesh, atom and basis metadata.
    ``factor_provider='streamed'`` contracts retained Bloch factors under a
    numerical-payload cap. ``'dense'`` selects the uncapped small-system oracle.
    """
    from ...dlpno.source import CorrelationSource, SourceConventions
    from ...dlpno.translation_builder import _record
    if type(compact_occupied) is not bool:
        raise ValueError('compact_occupied must be boolean')
    if factor_provider not in ('streamed', 'dense'):
        raise ValueError("factor_provider must be 'streamed' or 'dense'")
    if not isinstance(reference, ChiHFReference):
        raise TypeError('chi local MP2 requires a sealed ChiHFReference')
    if reference.method != 'RHF':
        raise NotImplementedError('chi local MP2 requires restricted RHF; UHF and ROHF are unsupported')
    hf, adapter, channels, frozen, _, executed, report = prepare_canonical_reference(
        reference, frozen_core=frozen_core)
    descriptor = adapter.operator.descriptor
    if descriptor.source != 'classic' or descriptor.engine != 'native-chi-classic-finite-contract':
        raise NotImplementedError('chi local MP2 requires the executed classic finite-contract source; SR is unsupported')
    domains = json.loads(descriptor.executed_domains_json)
    if domains.get('schema') != 'vibeqc.chi.fitted-domains/v1' or not isinstance(domains.get('controls'), dict):
        raise ValueError('chi local MP2 requires executed physical-pair controls')
    from .fitted import ChiCoulombControls
    require_public_mp2_controls('classic', ChiCoulombControls(**domains['controls']))
    physical = getattr(hf, '_chi_restart_physical', None)
    if not isinstance(physical, dict):
        raise ValueError('chi local MP2 requires sealed physical mesh/basis/geometry metadata')
    geometry = physical['geometry']
    if geometry['dimension'] != 3 or geometry['charge'] != 0 or geometry['multiplicity'] != 1:
        raise NotImplementedError('chi local MP2 requires a neutral 3D singlet')
    if (physical['options'] or {}).get('symmetry_reduce_k', False) or physical['controls'].get('symmetry_reduce_k', False):
        raise NotImplementedError('chi local MP2 does not admit symmetry-reduced references')
    symmetry = str(physical['controls'].get('symmetry_mode', 'off')).rsplit('.', 1)[-1].lower()
    if symmetry not in ('off', 'diagnostic'):
        raise NotImplementedError('chi local MP2 requires unreduced full-mesh integral execution')
    mesh = tuple(physical['mesh'])
    require_public_mp2_mesh(geometry['lattice_bohr'], mesh, hf.kpoints_cart, hf.kpoint_weights)
    if descriptor.orbital_basis_sha256 != physical['basis']['sha256']:
        raise ValueError('chi local MP2 reference and factor basis identities differ')
    convention = getattr(hf, 'finite_torus_convention', None)
    if (getattr(convention, 'exchange_q0', None) != 'bvk-ewald'
            or getattr(convention, 'coulomb_kernel', None) != '3d-periodic-g0'
            or tuple(getattr(convention, 'character_mesh_shape', ())) != mesh):
        raise ValueError('chi local MP2 requires the matched full-Coulomb BvK exchange convention')
    coeff, _, nocc = channels[0]
    nk, nao = coeff.shape[:2]
    if coeff.shape[2] != nao:
        raise NotImplementedError('chi local MP2 requires the full AO space; overlap-reduced references are unsupported')
    if frozen[0] >= nocc or physical['basis']['nbasis'] != nao:
        raise ValueError('chi local MP2 requires an active occupied space and matching AO metadata')
    from .ecp import electron_accounting
    counts = electron_accounting(hf, frozen, (nocc, nocc))
    if geometry['n_electrons'] != counts['physical_electrons']:
        raise ValueError('chi local MP2 physical electron count mismatch')
    workspace = None
    if compact_occupied:
        from ...dlpno.translation_preparation import selection_workspace
        workspace = selection_workspace(mesh, nao, nocc, max_selection_bytes)
    cells = tuple(product(*(range(n) for n in mesh)))
    phase = np.exp(2j*np.pi*(adapter.k_fractional @ np.asarray(cells).T))
    overlap, overlap_audit = _block_operator(np.asarray(hf.overlap), phase, mesh, 'overlap')
    fock, fock_audit = _block_operator(np.asarray(hf.fock), phase, mesh, 'Fock')
    if compact_occupied:
        from ...dlpno.translation_preparation import real_occupied_projector
        core, core_audit = real_occupied_projector(coeff[:, :, :frozen[0]], hf.overlap,
            phase, mesh, 'frozen', _REAL_TOL, kfrac=adapter.k_fractional)
        active, active_audit = real_occupied_projector(coeff[:, :, frozen[0]:nocc], hf.overlap,
            phase, mesh, 'active', _REAL_TOL, kfrac=adapter.k_fractional)
    else:
        workspace = None
        core, core_audit = _real_occupied(coeff[:, :, :frozen[0]], hf.overlap, phase, 'frozen')
        active, active_audit = _real_occupied(coeff[:, :, frozen[0]:nocc], hf.overlap, phase, 'active')
    if factor_provider == 'streamed':
        from ...dlpno.translation_bloch_factors import TranslationBlochFactors
        factors = TranslationBlochFactors(adapter, mesh, max_bytes=max_factor_bytes,
            auxiliary_tile=factor_auxiliary_tile, cache_bytes=factor_cache_bytes)
        factor_audit = dict(factors.audit)
    else:
        factors, factor_audit = _home_factors(adapter, phase, cells, mesh)
    molecule, basis = _layout(physical, nk)
    if compact_occupied:
        from ...dlpno.translation_projectors import BlochOccupiedProjectors, ProjectorReference
        projectors = BlochOccupiedProjectors(active, core, nocc*nk, int(frozen[0])*nk,
            overlap.identity, fock.identity,
            max(active_audit['orthogonality'], core_audit['orthogonality']),
            max(active_audit['projector_error'], core_audit['projector_error']),
            max(active_audit.get('imaginary', 0.), core_audit.get('imaginary', 0.)))
        torus = ProjectorReference(projectors, fock, overlap, np.asarray(channels[0][1]).ravel(),
            float(hf.energy)*nk, ecp_total_ncore=counts['ecp_core_electrons']*nk)
        occupied_bytes = active.blocks.nbytes+core.blocks.nbytes
    else:
        torus = SimpleNamespace(mo_coeffs=np.column_stack((core, active)), fock=fock,
            converged=True, energy=float(hf.energy)*nk, reference_kind='rhf',
            ecp_total_ncore=counts['ecp_core_electrons']*nk)
        occupied_bytes = torus.mo_coeffs.nbytes
    conventions = SourceConventions(exchange_q0=convention.exchange_q0, kmesh=mesh,
        bloch_normalisation='real-bvk-supercell', same_source=True)
    record = dict(kind='chi-translation-blocks-v1', source_identity=adapter.source_identity,
        reference_identity=reference.state_identity, request_identity=reference.request_identity,
        source=executed.as_dict(), retained_metric_ranks=adapter.retained_ranks,
        conventions=conventions.as_record(), scf_acceptance=dict(report),
        runtime=dict(hf.process_provenance),
        frozen_core_per_character=int(frozen[0]), n_frozen=int(frozen[0])*nk,
        electron_counts=counts, ecp=deepcopy(getattr(hf, 'chi_ecp', None)),
        overlap_identity=overlap.identity, fock_identity=fock.identity,
        factor_identity=factors.identity, real_transform_tolerance=_REAL_TOL,
        factor_provider=factor_provider, factor_provider_record=_record(factors.provenance),
        transform_audits=dict(overlap=overlap_audit, fock=fock_audit, factors=factor_audit,
                              active=active_audit, frozen=core_audit),
        method_convention='SCF Fock including BvK exchange zero mode; retained Coulomb ov factors',
        storage=dict(overlap_bytes=overlap.blocks.nbytes, fock_bytes=fock.blocks.nbytes,
                     home_factor_bytes=(factors.home_factors.nbytes if factor_provider == 'dense' else 0),
                     occupied_bytes=occupied_bytes, occupied_preparation=workspace,
                     scope='retained adapter arrays only; not a peak-memory estimate'))
    adapter.validate()
    reference.validate(reference.request_identity, reference.method)
    return CorrelationSource(kind='chi-translation-blocks', reference_kind='rhf',
        molecule=molecule, basis=basis, reference=torus, factors=factors,
        overlap=overlap, conventions=conventions, provenance=record)


@dataclass(frozen=True)
class ChiLocalMP2Result:
    """Raw iterated energy and separately identified semicanonical PNO estimate."""
    translation: object
    source_record_json: str
    settings_json: str
    adapter_identity: str
    method: str = 'translation-local-MP2'
    method_maturity: str = 'experimental'
    quantitative_status: str = 'not-qualified'
    backend: str = 'chi-translation-local-mp2'

    @property
    def raw_e_corr_per_cell(self):
        return self.translation.solution.e_pair_per_cell

    @property
    def pno_correction_per_cell(self):
        return self.translation.solution.e_pno_correction_per_cell

    @property
    def e_corr_per_cell(self):
        return self.translation.solution.e_corr_per_cell

    @property
    def e_total_per_cell(self):
        return self.translation.solution.e_tot_per_cell

    @property
    def converged(self):
        return self.translation.solution.converged


def run_chi_local_mp2(reference, *, frozen_core=0, tcut_mkn=0., tcut_pno=0.,
                      lindep=1e-8, seed_lindep=1e-8, tolerance=1e-10,
                      max_iter=100, conv_tol_energy=1e-9, conv_tol_residual=1e-7,
                      damping=1., max_transport_bytes=256*1024**2,
                      factor_provider='streamed', max_factor_bytes=64*1024**2,
                      factor_auxiliary_tile=16, factor_cache_bytes=1024**2,
                      compact_storage=True, max_selection_bytes=64*1024**2, output=None):
    """Explicit experimental library route using the shared iterative solver.

    Zero cutoffs retain full domains/PNOs. ``max_transport_bytes`` limits
    prepared native transport storage/workspace only, not SCF, adapter,
    PAO/PNO preparation, Python metadata or the process RSS. Nonconvergence
    is returned explicitly; corrections never enter the residual test.
    ``max_factor_bytes`` separately caps streamed factor audit/contraction
    arrays, phase cache and returned buffers until return. Pre-existing
    source/caller arrays, reference admission and downstream retained outputs
    are excluded. Auxiliary tiles shrink to fit, or fail before allocation.
    The explicit dense factor oracle does not use this cap.
    ``compact_storage=True`` retains selected occupied home columns and
    projected pair recipes, reconstructing individual pair AO bases for
    complete validation and contractions. False retains the dense oracle.
    ``max_selection_bytes`` preflights controlled numerical payload for the
    compact occupied-source audit, selector and its complete admission checks.
    Native library internals, pre-existing inputs, and downstream pair work
    are excluded; this is not a total-memory bound.
    ``output`` optionally writes .out/.system and citation siblings through
    vibeqc.output. Without it, active output/structured channels receive the
    same record. No correlation restart format is implied.
    """
    from ...dlpno.translation_driver import run_translation_dlpno_mp2
    source = build_chi_translation_source(reference, frozen_core=frozen_core,
        factor_provider=factor_provider, max_factor_bytes=max_factor_bytes,
        factor_auxiliary_tile=factor_auxiliary_tile, factor_cache_bytes=factor_cache_bytes,
        compact_occupied=compact_storage, max_selection_bytes=max_selection_bytes)
    settings = dict(tcut_mkn=tcut_mkn, tcut_pno=tcut_pno, pno_norm='mp2',
        lindep=lindep, seed_lindep=seed_lindep, tolerance=tolerance, max_iter=max_iter,
        conv_tol_energy=conv_tol_energy, conv_tol_residual=conv_tol_residual, damping=damping,
        max_transport_bytes=max_transport_bytes, factor_audit='auto', transport_backend='prepared',
        compact_storage=compact_storage, max_selection_bytes=max_selection_bytes)
    result = run_translation_dlpno_mp2(source, source.conventions.kmesh,
        n_frozen=source.provenance['n_frozen'], **settings)
    settings.update(factor_provider=factor_provider, max_factor_bytes=max_factor_bytes,
        factor_auxiliary_tile=factor_auxiliary_tile, factor_cache_bytes=factor_cache_bytes)
    # Include actual contraction/cache high-water accounting after pair work.
    from ...dlpno.translation_builder import _record
    source_record = deepcopy(source.provenance)
    source_record['factor_provider_execution'] = _record(source.factors.provenance)
    reference.validate(reference.request_identity, reference.method)
    result = ChiLocalMP2Result(result, json.dumps(source_record, sort_keys=True),
                              json.dumps(settings, sort_keys=True), source.identity)
    from ...output.chi_correlation import emit_chi_local_mp2
    emit_chi_local_mp2(result, output=output)
    return result
