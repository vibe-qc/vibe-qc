"""Qualified molecular RI factors with complex Abelian block contractions.

Melega et al., doi:10.1021/acs.jpca.5c08579, Eqs. (3)-(5), supplies the
fitted-factor identity, not derivative admission. Kitsaras and Stopkowicz,
arXiv:2602.12457v1, Eqs. (16)-(17), supplies conjugated irrep products.
MANTA owns the integrals and metric factorization. This bounded consumer
snapshots their finite source once and never changes its retained rank.
"""
from dataclasses import dataclass, field, replace
from hashlib import sha256
from itertools import product
import json
import math
from time import perf_counter

import numpy as np

from . import _vibeqc_core as core
from .symmetry_abelian import AbelianCharacters
from .symmetry_ao import build_molecular_space_action, molecular_point_group
from .symmetry_shared import (Budget, SpaceIdentity, OperatorContract,
    _array, _freeze, _integer, _tolerance, audit_subspace_panels)
from .symmetry_quartets import _norm, _shell_signature, QuartetWork
from .symmetry_quartets_hierarchy import _native_budget
from .symmetry_molecular_hf import _MolecularJKCapture


def _adapt_retained(irreps, actions, metric, coefficients, tolerance, budget,
        *, preserve_native_gram=False):
    """Rotate within exactly the supplied subspace, without refactoring it.

    Native RI factors retain their actual finite Gram matrix. A QR coordinate
    solve measures containment without a second ill-conditioned Coulomb
    metric solve. It neither replaces the factor nor changes its rank cut.
    Ordinary orthonormal panels keep the stricter shared audit contract.
    """
    metric = np.ascontiguousarray(metric, dtype=complex)
    coefficients = np.ascontiguousarray(coefficients, dtype=complex)
    rank = coefficients.shape[1]
    if rank == 0:
        raise ValueError('empty RI space; use the unreduced producer')
    contract = OperatorContract('finite metric and retained space', 'owned snapshot',
        'all supplied columns', 'unchanged retained rank', 'coefficient transport')
    space = actions[0].source
    retained = SpaceIdentity(space.geometry, space.basis, 'retained columns', space.gauge)
    gram = coefficients.conj().T@metric@coefficients
    gram_defect = _norm(gram-np.eye(rank))
    if preserve_native_gram:
        if (not np.isfinite(gram).all() or _norm(gram-gram.conj().T) > tolerance
                or np.min(np.linalg.eigvalsh(gram)) <= 0):
            raise ValueError('invalid retained native factor metric')
        q, triangular = np.linalg.qr(coefficients, mode='reduced')
    matrices, leakage, metric_residual = [], 0., 0.
    for action in actions:
        if action.antiunitary:
            raise ValueError('antiunitary actions need separate RI admission')
        residual = _norm(action.pull_operator(metric)-metric)
        if residual > tolerance:
            raise ValueError('RI metric is not equivariant; use the unreduced producer')
        if preserve_native_gram:
            moved = action.apply(coefficients)
            mixing = np.linalg.solve(triangular, q.conj().T@moved)
            contained = _norm(moved-coefficients@mixing)/_norm(moved)
            isometry = _norm(moved.conj().T@metric@moved-gram)
            if not math.isfinite(contained+isometry) or max(contained, isometry) > tolerance:
                raise ValueError('retained RI space is not closed or isometric; use the unreduced producer')
        else:
            evidence = audit_subspace_panels(action, coefficients, coefficients, metric, metric,
                source_subspace=retained, target_subspace=retained, contract=contract,
                metric_tolerance=tolerance, leakage_tolerance=tolerance,
                probe_identity='complete retained RI panel', budget=budget)
            if not evidence.passed_probe:
                raise ValueError('retained RI space is not closed; use the unreduced producer')
            mixing, contained, isometry = evidence.mixing, evidence.containment.residual, evidence.metric.residual
        matrices.append(mixing)
        leakage = max(leakage, contained)
        metric_residual = max(metric_residual, residual, isometry)
    rotations, labels = [], []
    for label, characters in enumerate(irreps.characters):
        projector = sum(c.conjugate()*u for c, u in zip(characters, matrices))/len(matrices)
        if _norm(projector-projector.conj().T) > tolerance:
            raise ValueError('non-Hermitian irrep projector; use the unreduced producer')
        # eigh's lower triangle chooses a basis only. All projector and
        # intertwining relations are checked against the original matrices.
        values, vectors = np.linalg.eigh(projector)
        if np.max(np.minimum(abs(values), abs(values-1)), initial=0.) > tolerance:
            raise ValueError('irrep projector is not idempotent')
        selected = vectors[:, values > .5]
        if _norm(projector@selected-selected) > tolerance:
            raise ValueError('invalid irrep projector range')
        rotations.append(selected)
        labels.extend([label]*selected.shape[1])
    rotation = np.column_stack(rotations)
    if rotation.shape != (rank, rank) or _norm(rotation.conj().T@rotation-np.eye(rank)) > tolerance:
        raise ValueError('irrep spaces do not resolve the complete retained rank')
    if preserve_native_gram:
        # Polar cleanup rotates only factor coordinates, never whitens F or
        # changes F F^dagger. Intertwining and the final source error remain
        # checked at the unchanged tolerance after this finite correction.
        left, _, right = np.linalg.svd(rotation, full_matrices=False)
        rotation = left@right
    adapted = coefficients@rotation
    for g, action in enumerate(actions):
        delta = action.apply(np.ascontiguousarray(adapted))-adapted*irreps.characters[labels, g]
        if _norm(delta) > tolerance*max(1., _norm(adapted)):
            raise ValueError('retained partner functions fail the irrep action')
    return _freeze(adapted), _freeze(np.array(labels, np.int64)), dict(
        rank=rank, maximum_metric_residual=metric_residual, maximum_relative_leakage=leakage,
        metric_gram_identity_residual=gram_defect, preserved_native_gram=preserve_native_gram), rotation


@dataclass(frozen=True, slots=True)
class RIFactorJKResult:
    j: np.ndarray | None
    k: np.ndarray | None
    scalar_products: int
    block_contractions: int
    transport_scalar_products: int
    error_bound: float


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AbelianMolecularRI:
    """Owned finite MANTA RI source; factory-only, no serializable authority.

    The snapshot uses all ordered three-center tiles and the producer's exact
    positive metric factor. Only complete retained spaces may be adapted.
    Complex auxiliary partners use E[p,q,r,s] = sum_A B[A,p,q] B[A,s,r]*.
    This source admits neither periodic kernels nor analytic derivatives.
    """
    irreps: AbelianCharacters
    orbital_coefficients: np.ndarray
    orbital_inverse: np.ndarray
    auxiliary_coefficients: np.ndarray
    orbital_labels: np.ndarray
    auxiliary_labels: np.ndarray
    indices: tuple
    auxiliary_indices: tuple
    blocks: tuple
    source_identity: str
    source_tolerance: float
    tensor_error_bound: float
    required_bytes: int
    required_work: int
    _receipt_json: str = field(repr=False)
    _native: object = field(repr=False)
    _block_executor: object = field(repr=False)
    _block_plan: object = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError('use prepare_abelian_molecular_ri for a native source')

    @property
    def n(self):
        return len(self.orbital_labels)

    @property
    def citation_numerics(self):
        return ('symmetry_molecular_ri',)

    @property
    def receipt(self):
        return json.loads(self._receipt_json)

    def requirements(self, *, targets='jk', executor='blocked', budget):
        if targets not in ('j', 'k', 'jk') or executor not in ('blocked', 'dense', 'numpy'):
            raise ValueError('explicit RI J/K targets and blocked/dense/numpy executor required')
        rank = len(self.auxiliary_labels)
        size = self.required_bytes+65536+256*self.n**2+32*rank*self.n**2
        # Preparation has already completed on this immutable source. Bound
        # only executed contraction/transport and block traversal here;
        # required_work in the source receipt still reports preparation.
        work = 128*(rank+1)*self.n**3+128*len(self.auxiliary_indices)*(len(self.blocks)+1)
        work = max(work, self._block_plan.work+128*self.n**3)
        budget.admit(size, work)
        return Budget(size, work)

    def jk(self, density, *, budget, targets='jk', executor='blocked'):
        """Arbitrary complex AO D, including nonsymmetric transition densities.

        D_adapted = C^-1 D C^-dagger, F_AO = C^-dagger F_adapted C^-1.
        J = sum B tr(B^dagger D); K = sum B D B^dagger. In particular, D
        is never conjugated or projected onto an invariant-state block.
        Counts denote executed complex scalar products, not real FLOPs.
        """
        self.requirements(targets=targets, executor=executor, budget=budget)
        _array(density, 'complex128', 2)
        if density.shape != (self.n, self.n) or not np.isfinite(density).all():
            raise ValueError('invalid molecular RI density')
        inverse = self.orbital_inverse
        d = inverse@density@inverse.conj().T
        j = np.zeros_like(d) if 'j' in targets else None
        k = np.zeros_like(d) if 'k' in targets else None
        terms = contractions = 0
        if executor == 'dense':
            # Independent dense fallback on the identical selected factors.
            b = np.zeros((len(self.auxiliary_labels), self.n, self.n), complex)
            for (a, q, c), block in self.blocks:
                b[np.ix_(self.auxiliary_indices[c], self.indices[a], self.indices[q])] = block
            if j is not None:
                rho = np.einsum('Apq,pq->A', b.conj(), d, optimize=False)
                j = np.einsum('A,Apq->pq', rho, b, optimize=False)
                terms += 2*b.size; contractions += 2
            if k is not None:
                temporary = np.einsum('Apq,qr->Apr', b, d, optimize=False)
                k = np.einsum('Apr,Asr->ps', temporary, b.conj(), optimize=False)
                terms += 2*len(b)*self.n**3; contractions += 2
        elif executor == 'blocked':
            j, k, counts = self._block_executor.contract(np.ascontiguousarray(d),
                j is not None, k is not None, budget.native())
            terms, contractions = counts['scalar_products'], counts['block_contractions']
        else:
            for c, aux in enumerate(self.auxiliary_indices):
                group = [(a, q, block) for (a, q, label), block in self.blocks if label == c]
                if j is not None and group:
                    rho = np.zeros(len(aux), complex)
                    for a, q, block in group:
                        rho += np.einsum('Apq,pq->A', block.conj(), d[np.ix_(self.indices[a], self.indices[q])], optimize=False)
                        terms += block.size; contractions += 1
                    for a, q, block in group:
                        j[np.ix_(self.indices[a], self.indices[q])] += np.einsum('A,Apq->pq', rho, block, optimize=False)
                        terms += block.size; contractions += 1
                if k is not None:
                    for a, q, left in group:
                        for s, r, right in group:
                            middle = d[np.ix_(self.indices[q], self.indices[r])]
                            temporary = np.einsum('Apq,qr->Apr', left, middle, optimize=False)
                            k[np.ix_(self.indices[a], self.indices[s])] += np.einsum('Apr,Asr->ps', temporary, right.conj(), optimize=False)
                            terms += len(aux)*len(self.indices[a])*len(self.indices[r])*(len(self.indices[q])+len(self.indices[s]))
                            contractions += 2
        result = [None if f is None else inverse.conj().T@f@inverse for f in (j, k)]
        if any(f is not None and not np.isfinite(f).all() for f in result):
            raise ValueError('nonfinite RI contraction')
        error = self.tensor_error_bound*_norm(density)
        if not math.isfinite(error) or (error == 0 and self.tensor_error_bound > 0 and _norm(density) > 0):
            raise ValueError('nonrepresentable RI error bound')
        return RIFactorJKResult(*(None if f is None else _freeze(f) for f in result),
            terms, contractions, 2*self.n**3*(1+len(targets)), error)


def prepare_abelian_molecular_ri(molecule, orbital, auxiliary, rotations, translations,
        characters, *, source_tolerance, budget: Budget, ri_options=None, factor_export='producer'):
    """Qualify an all-space real molecular Coulomb RI source for block J/K.

    Full positive AO overlap is required for this reversible coordinate
    change. The method still owns its later occupied/virtual and overlap rank
    choices. Auxiliary rank is selected only by native RIOptions. No primitive
    screening, new kernel, range separation or metric regularization is added.
    """
    tolerance = _tolerance(source_tolerance, 'source tolerance')
    if factor_export not in ('producer', 'ordered-tiles'):
        raise ValueError('factor_export must be producer or ordered-tiles')
    if tolerance > 1e-8:
        raise ValueError('source tolerance must not exceed 1e-8')
    _array(rotations, 'float64', 3); _array(translations, 'float64', 2)
    _array(characters, 'complex128', 2)
    n, m, ng = orbital.nbasis, auxiliary.nbasis, len(rotations)
    if min(n, m, ng) < 1 or rotations.shape != (ng, 3, 3) or translations.shape != (ng, 3) or characters.shape != (ng, ng):
        raise ValueError('invalid molecular RI dimensions or operations')
    # Admission before basis snapshots, native metric construction, payload
    # copies or eigensolves. Count all coexisting dense factor/audit arrays.
    size = 262144+512*(m*n*n+ng*(n*n+m*m)+m*m+n*n)+1024*ng**3
    work = 4096*(m*m*m+ng*(n**3+m**3)+m*n**3+m*m*n*n+ng**4)
    budget.admit(size, work)
    native_budget = _native_budget(Budget(budget.maximum_bytes-size, budget.maximum_work))
    if ri_options is not None and type(ri_options) is not core._manta.RIOptions:
        raise TypeError('native RIOptions required')
    options = core._manta.RIOptions()
    if ri_options is not None:
        for name in ('absolute_cutoff', 'relative_cutoff', 'negative_relative_tolerance', 'mode_block'):
            setattr(options, name, getattr(ri_options, name))
    request = core._manta.KernelRequest()
    native_plan = core._manta.plan_ri(orbital, auxiliary, options, request, native_budget)
    fitting_plan = core._manta.plan_fitting([auxiliary, orbital, orbital], native_budget)
    python_bytes = size
    size += native_plan.modeled_peak_bytes+(fitting_plan.modeled_peak_bytes if factor_export == 'ordered-tiles' else 0)
    work += native_plan.kernel_terms+fitting_plan.kernel_terms
    budget.admit(size, work)
    signatures = tuple(_shell_signature(b.shells()) for b in (orbital, auxiliary))
    started = perf_counter()
    group = molecular_point_group(molecule, rotations, translations,
        identity_label='molecular RI operations', budget=budget)
    irreps = AbelianCharacters(group, characters, budget=budget, tolerance=tolerance)
    native = core._manta.RIJKBuilder(orbital, auxiliary, options, request, native_budget)
    if factor_export == 'producer':
        export_plan = native.plan_factor_export(native_budget)
        size = python_bytes+export_plan.modeled_peak_bytes
        work += export_plan.kernel_terms
        budget.admit(size, work)
    else:
        provider = core._manta.FittingTiles([auxiliary, orbital, orbital], native_budget)
    if signatures != tuple(_shell_signature(b.shells()) for b in (orbital, auxiliary)):
        raise ValueError('RI basis changed during source construction')
    # The producer explicitly uses the lower triangle. Do not average it or
    # recompute the rank/eigenvectors in another factorization.
    raw_metric = np.asarray(native.metric)
    metric = np.tril(raw_metric)+np.tril(raw_metric, -1).T
    factor = np.asarray(native.factor)
    overlap = np.asarray(core.compute_overlap(orbital))
    values, vectors = np.linalg.eigh(overlap)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('nonpositive AO overlap; use the unreduced RI producer')
    orbital_space = SpaceIdentity('finite molecule', native.source_identity, 'orbital AO', 'native overlap')
    auxiliary_space = SpaceIdentity('finite molecule', native.source_identity, 'auxiliary AO', 'Coulomb metric')
    actions = tuple(tuple(build_molecular_space_action(molecule, b, r, t,
        source=space, target=space, budget=budget) for r, t in zip(rotations, translations))
        for b, space in ((orbital, orbital_space), (auxiliary, auxiliary_space)))
    coefficients, labels, ao_evidence, _ = _adapt_retained(irreps, actions[0], overlap,
        vectors/np.sqrt(values), tolerance, budget)
    aux_coefficients, aux_labels, aux_evidence, aux_rotation = _adapt_retained(irreps, actions[1], metric,
        factor, tolerance, budget, preserve_native_gram=True)
    # Use the actual finite change of coordinates, not the identity C^-1=C^H S
    # that holds only for exactly metric-orthonormal columns.
    inverse = np.linalg.solve(coefficients, np.eye(n))
    # Auxiliary coefficient is unconjugated in (p* q | A). RI reconstruction
    # conjugates the second factor and reverses its AO pair, not its density.
    export_receipt = None
    if factor_export == 'producer':
        exported = native.export_factors(native_budget)
        evidence = exported.receipt
        if (type(exported) is not core._manta.RIFactorExport
                or evidence.source_identity != native.source_identity
                or evidence.metric_factor_identity != native.metric_info.factor_identity
                or evidence.n_basis != n or evidence.retained_rank != len(aux_labels)
                or evidence.kernel != 'coulomb' or evidence.omega != 0.
                or evidence.provider != 'manta' or evidence.provider_version != 'md-fp64-v1'
                or evidence.algorithm != 'ordered-ri-factor-export-v1'
                or evidence.normalization != 'libint-shell-axial-v1'
                or evidence.raw_coupling_array_bytes != 0):
            raise ValueError('changed or incompatible native RI factor export')
        export_receipt = {name: getattr(evidence, name) for name in (
            'provider', 'provider_version', 'algorithm', 'normalization', 'kernel', 'omega',
            'source_identity', 'coupling_identity', 'metric_factor_identity', 'export_identity',
            'n_basis', 'retained_rank', 'shell_calls', 'primitive_products', 'integral_elements_consumed',
            'contraction_products', 'accumulation_terms', 'factor_array_bytes', 'raw_coupling_array_bytes',
            'modeled_peak_bytes')}
        original_ao = np.asarray(exported.coefficients).reshape(len(aux_labels), n, n)
        del exported
        if (not np.isfinite(original_ao).all() or evidence.shell_calls != fitting_plan.shell_calls
                or evidence.primitive_products != export_plan.primitive_products
                or evidence.integral_elements_consumed != export_plan.integral_elements
                or evidence.contraction_products != export_plan.contraction_products
                or evidence.accumulation_terms != export_plan.accumulation_terms):
            raise ValueError('RI factor export census failed')
        full_ao = np.einsum('BA,Buv->Auv', aux_rotation, original_ao, optimize=True)
        coupling_identity = evidence.coupling_identity
        calls, primitive_products = evidence.shell_calls, evidence.primitive_products
        raw_bytes = 0
    else:
        coupling = np.empty((m, n, n))
        offsets = [actions[1][0].offsets, actions[0][0].offsets, actions[0][0].offsets]
        calls = primitive_products = 0
        for key in product(*(range(len(o)-1) for o in offsets)):
            tile = provider.evaluate(key)
            if (tile['source_identity'] != provider.source_identity or tile['kernel'] != 'coulomb'
                    or tile['omega'] != 0. or tile['provider'] != 'manta'
                    or tile['normalization'] != 'libint-shell-axial-v1'):
                raise ValueError('changed or incompatible three-center source')
            slices = tuple(slice(int(o[k]), int(o[k+1])) for o, k in zip(offsets, key))
            coupling[slices] = tile['values']
            calls += tile['shell_calls']; primitive_products += tile['primitive_products_requested']
        if calls != fitting_plan.shell_calls or not np.isfinite(coupling).all():
            raise ValueError('RI coupling census failed')
        full_ao = np.einsum('PA,Puv->Auv', aux_coefficients, coupling, optimize=True)
        original_ao = np.einsum('PA,Puv->Auv', factor, coupling, optimize=True)
        coupling_identity, raw_bytes = provider.source_identity, coupling.nbytes
    adapted = np.einsum('up,Auv,vq->Apq', coefficients.conj(), full_ao, coefficients, optimize=True)
    indices = tuple(_freeze(np.flatnonzero(labels == a)) for a in range(ng))
    aux_indices = tuple(_freeze(np.flatnonzero(aux_labels == a)) for a in range(ng))
    blocks, omitted = [], adapted.copy()
    for a, q, c in product(range(ng), repeat=3):
        if not len(indices[a]) or not len(indices[q]) or not len(aux_indices[c]):
            continue
        selected = np.ix_(aux_indices[c], indices[a], indices[q])
        if irreps.allows((a, q, c), (-1, 1, 1)):
            blocks.append(((a, q, c), _freeze(adapted[selected])))
            omitted[selected] = 0.
    omitted_ao = np.einsum('pu,Apq,qv->Auv', inverse.conj(), omitted, inverse, optimize=True)
    selected_ao = np.einsum('pu,Apq,qv->Auv', inverse.conj(), adapted-omitted, inverse, optimize=True)
    delta, norm = _norm(selected_ao-full_ao), _norm(full_ao)
    gauge_defect = _norm(aux_rotation@aux_rotation.conj().T-np.eye(len(aux_labels)))
    gauge_error = _norm(original_ao)**2*gauge_defect
    error = delta*(2*norm+delta)+gauge_error
    if (not math.isfinite(error) or error > tolerance
            or (error == 0 and delta > 0 and norm > 0)):
        raise ValueError('fitted source violates symmetry selection; use the unreduced RI producer')
    if signatures != tuple(_shell_signature(b.shells()) for b in (orbital, auxiliary)):
        raise ValueError('RI basis changed during source qualification')
    # Compile only the already qualified selected tensor. This executor owns
    # its copy but carries no producer, group, state or screening authority.
    # Adaptation above concatenates complete irreps, so these partitions use
    # precisely the ordering of the immutable public factor blocks.
    elements = sum(b.size for _, b in blocks)
    block_plan = core._plan_symmetry_ri_blocks(n, len(aux_labels), ng,
        len(blocks), elements, budget.native())
    size += block_plan.bytes
    work += block_plan.work
    budget.admit(size, work)
    if (not np.array_equal(np.concatenate(indices), np.arange(n)) or
            not np.array_equal(np.concatenate(aux_indices), np.arange(len(aux_labels)))):
        raise ValueError('native RI blocks require contiguous complete irrep coordinates')
    orbital_offsets = np.array([0, *np.cumsum([len(i) for i in indices])], np.int64)
    auxiliary_offsets = np.array([0, *np.cumsum([len(i) for i in aux_indices])], np.int64)
    keys = np.asarray([key for key, _ in blocks], np.int64).reshape(-1, 3)
    packed = np.concatenate([b.ravel() for _, b in blocks]) if blocks else np.empty(0, complex)
    block_executor = core._SymmetryRIBlocks(n, len(aux_labels), orbital_offsets,
        auxiliary_offsets, keys, packed, budget.native())
    del packed
    digest = sha256(b'vibeqc-abelian-molecular-ri-v1\0')
    digest.update(native.source_identity.encode()); digest.update(coupling_identity.encode())
    if export_receipt is not None:
        digest.update(export_receipt['export_identity'].encode())
    for array in (coefficients, inverse, aux_coefficients, labels, aux_labels):
        digest.update(array.tobytes())
    for key, block in blocks:
        digest.update(np.asarray(key, np.int64).tobytes()); digest.update(block.tobytes())
    receipt = dict(schema='vibeqc-abelian-molecular-ri-v1', source_identity=digest.hexdigest(),
        producer_identity=native.source_identity, coupling_identity=coupling_identity,
        metric_factor_identity=native.metric_info.factor_identity,
        metric_rank=native.metric_info.rank, metric_cutoff=native.metric_info.cutoff,
        kernel='coulomb', finite_support='all ordered molecular three-center tiles',
        primitive_precision=0., orbital_evidence=ao_evidence, auxiliary_evidence=aux_evidence,
        source_tolerance=tolerance, omitted_factor_norm_AO=_norm(omitted_ao),
        factor_reconstruction_norm_AO=delta, auxiliary_gauge_defect=gauge_defect,
        auxiliary_gauge_error_bound=gauge_error, tensor_error_bound=error,
        arithmetic_roundoff_included=False, stored_factor_elements=sum(b.size for _, b in blocks),
        dense_factor_elements=adapted.size, factor_array_bytes=sum(b.nbytes for _, b in blocks),
        orbital_transform_bytes=coefficients.nbytes+inverse.nbytes,
        auxiliary_transform_bytes=aux_coefficients.nbytes,
        native_metric_factor_array_bytes=raw_metric.nbytes+factor.nbytes,
        overlap_build_calls=1, metric_shell_calls=native.metric_info.shell_calls, coupling_shell_calls=calls,
        metric_primitive_products=native.metric_info.primitive_products,
        coupling_primitive_products=primitive_products,
        factor_export=factor_export, factor_export_receipt=export_receipt,
        block_executor='native-auxiliary-panel-v1',
        native_factor_copy_bytes=16*elements, native_block_plan_bytes=block_plan.bytes,
        raw_coupling_array_bytes=raw_bytes,
        required_bytes=size, required_work=work, setup_seconds=perf_counter()-started)
    result = object.__new__(AbelianMolecularRI)
    for name, value in dict(irreps=irreps, orbital_coefficients=coefficients,
        orbital_inverse=_freeze(inverse), auxiliary_coefficients=aux_coefficients,
        orbital_labels=labels, auxiliary_labels=aux_labels, indices=indices,
        auxiliary_indices=aux_indices, blocks=tuple(blocks), source_identity=digest.hexdigest(),
        source_tolerance=tolerance, tensor_error_bound=error, required_bytes=size,
        required_work=work, _receipt_json=json.dumps(receipt, allow_nan=False), _native=native,
        _block_executor=block_executor, _block_plan=block_plan).items():
        object.__setattr__(result, name, value)
    return result


@dataclass(frozen=True)
class MolecularRIRouteReceipt:
    """Bounded scalar cost evidence, never source or state authority."""
    source_identity: str
    selected: str
    reason: str
    expected_builds: int = 0
    calibration_seconds: float = 0.
    maximum_difference: float = 0.
    samples: tuple = ()
    logical_bytes: int = 0
    modeled_work: int = 0
    targets: tuple[str, ...] = ()


class MolecularRIJK(_MolecularJKCapture, core.JKBuilder):
    """Explicit real-AO RHF/UHF bridge; shared FockBuild owns evaluated state.

    The factors may be complex even though the physical molecular AO source
    and native method densities are real. Imaginary residuals are checked,
    never silently discarded or repaired by Hermitization. Native primitive
    screening and incremental-density state do not modify this fixed source.
    """
    def __init__(self, source, *, budget: Budget, executor='blocked', capture_terminal=False):
        super().__init__()
        if type(source) is not AbelianMolecularRI:
            raise TypeError('an owned AbelianMolecularRI source is required')
        if executor not in ('dense', 'blocked', 'auto'):
            raise ValueError('RI executor must be dense, blocked or auto')
        self._source, self._budget, self._executor = source, budget, executor
        self._routes = {}
        self._route_request = None
        self._initialize_capture(capture_terminal)
        self.requirements(('jk',))
        from .symmetry_molecular_hf import MolecularJKTotals
        self._totals, self._last_build = MolecularJKTotals(), None

    @property
    def source(self):
        return self._source

    @property
    def n(self):
        return self.source.n

    @property
    def totals(self):
        return self._totals

    @property
    def last_build(self):
        return self._last_build

    @property
    def route_calibrations(self):
        return tuple(self._routes.items())

    def fork(self, *, capture_terminal=False):
        """Same immutable source and scalar profiles, fresh physical counters."""
        result = MolecularRIJK(self.source, budget=self._budget, executor=self._executor,
            capture_terminal=capture_terminal)
        result._routes = dict(self._routes)
        return result

    def requirements(self, targets):
        from .symmetry_quartets_batch import _target_counts
        nj, nk = _target_counts(len(targets), targets)
        if not targets:
            raise ValueError('at least one RI density is required')
        # Both executors have the same conservative reservation. Auto must
        # remain capable of its explicit dense fallback before reading D.
        executor = 'dense' if self._executor == 'auto' else self._executor
        base = self.source.requirements(executor=executor, budget=self._budget)
        size = base.maximum_bytes+65536+64*(len(targets)+nj+nk)*self.n**2
        if self._executor == 'auto':
            size += 131072  # five signatures, at most ten scalar samples each
        capture_bytes, capture_work = self._capture_requirements()
        size += capture_bytes
        work = base.maximum_work*len(targets)+capture_work
        self._budget.admit(size, work)
        return Budget(size, work)

    def _qualified_source_identity(self):
        return self.source.source_identity

    def _density(self, density):
        from .symmetry_molecular_hf import ScreenedMolecularJK
        return ScreenedMolecularJK._density(self, density)

    def calibrate_routes(self, *, targets, expected_builds, repetitions=2):
        """Compare fixed nonsymmetric probes on the identical selected source.

        Unmeasured signatures use dense factors. Cost includes all paired
        probes; estimates that cannot repay even baseline probes skip them.
        No integral, metric, density or factor-mask change occurs here.
        """
        from .symmetry_molecular_hf import _ROUTE_SIGNATURES, _calibration_cannot_pay, _select_route
        if self._executor != 'auto':
            raise ValueError('route calibration requires executor="auto"')
        if targets not in _ROUTE_SIGNATURES:
            raise ValueError('calibration requires a native HF channel signature')
        expected_builds = _integer(expected_builds, 'expected builds', positive=True)
        repetitions = _integer(repetitions, 'repetitions', positive=True)
        if repetitions > 5:
            raise ValueError('at most five paired route probes are admitted')
        reservation = self.requirements(targets)
        if _calibration_cannot_pay(expected_builds, repetitions):
            route = MolecularRIRouteReceipt(self.source.source_identity, 'dense',
                'probe cost cannot repay within expected builds', expected_builds,
                logical_bytes=reservation.maximum_bytes, targets=targets)
            self._routes[targets] = route
            return route
        size = reservation.maximum_bytes+65536+64*(len(targets)+sum(map(len, targets)))*self.n**2
        work = 2*repetitions*reservation.maximum_work+128*sum(map(len, targets))*self.n**2
        self._budget.admit(size, work)
        self._routes.pop(targets, None)
        started = perf_counter()
        rng = np.random.default_rng(64129)
        densities = tuple(rng.uniform(-1., 1., (self.n, self.n)) for _ in targets)
        samples, times, difference = [], {'dense': [], 'blocked': []}, 0.
        for repeat in range(repetitions):
            order = ('dense', 'blocked') if repeat % 2 == 0 else ('blocked', 'dense')
            results = {}
            for executor in order:
                j, k, receipt = self._execute(densities, targets, executor)
                results[executor] = (j, k)
                samples.append(receipt)
                times[executor].append(receipt.seconds)
            for family in (0, 1):
                for a, b in zip(results['dense'][family], results['blocked'][family]):
                    delta = _norm(a-b)
                    difference = max(difference, delta)
                    # Same factors/mask, so no source-omission allowance is
                    # added to this unchanged arithmetic comparison gate.
                    if not math.isfinite(delta) or delta > self.source.source_tolerance:
                        raise ValueError('RI route probe disagrees on the identical selected source')
            del results
        elapsed = perf_counter()-started
        selected, reason = _select_route(times['dense'], times['blocked'], elapsed,
            expected_builds, baseline='dense', candidate='blocked')
        route = MolecularRIRouteReceipt(self.source.source_identity, selected, reason,
            expected_builds, elapsed, difference, tuple(samples), size, work, targets)
        self._routes[targets] = route
        return route

    def build_channels(self, densities, *, targets):
        from .symmetry_molecular_hf import MolecularJKTotals
        capture = self._begin_capture(targets)
        self.requirements(targets)
        if self._executor == 'auto':
            if self._route_request is not None:
                self._route_request(self, densities, targets)
            route = self._routes.get(targets, MolecularRIRouteReceipt(self.source.source_identity,
                'dense', 'no measured calibration for these channels'))
        else:
            route = MolecularRIRouteReceipt(self.source.source_identity, self._executor, 'explicit executor')
        j, k, receipt = self._execute(densities, targets, route.selected)
        receipt = replace(receipt, route=route)
        old = self._totals
        self._totals = MolecularJKTotals(old.builds+1, old.work,
            old.scalar_products+receipt.scalar_products, old.block_contractions+receipt.block_contractions,
            old.transport_scalar_products+receipt.transport_scalar_products, old.seconds+receipt.seconds,
            max(old.maximum_logical_bytes, receipt.logical_bytes))
        self._last_build = receipt
        if self._capture_terminal:
            self._capture(densities, j, k, receipt, capture)
        return j, k, receipt

    def _execute(self, densities, targets, executor):
        from .symmetry_molecular_hf import MolecularJKReceipt
        memory = self.requirements(targets)
        if not isinstance(densities, tuple) or len(densities) != len(targets):
            raise ValueError('RI density/channel dimensions disagree')
        values = tuple(self._density(d) for d in densities)
        started = perf_counter()
        js, ks, bounds = [], [], []
        terms = contractions = transport = 0
        for density, target in zip(values, targets):
            result = self.source.jk(np.ascontiguousarray(density, dtype=complex),
                budget=self._budget, targets=target, executor=executor)
            for value, output in ((result.j, js), (result.k, ks)):
                if value is not None:
                    if _norm(value.imag) > self.source.source_tolerance+result.error_bound:
                        raise ValueError('real molecular RI source returned a complex operator')
                    output.append(np.array(value.real, copy=True))
            bounds.append(result.error_bound)
            terms += result.scalar_products
            contractions += result.block_contractions
            transport += result.transport_scalar_products
        receipt = MolecularJKReceipt(self.source.source_identity, targets, tuple(bounds),
            QuartetWork(), terms, contractions, transport, perf_counter()-started,
            memory.maximum_bytes, executor='ri-'+executor)
        return (np.array(js).reshape(-1, self.n, self.n),
                np.array(ks).reshape(-1, self.n, self.n), receipt)

    def build_J(self, density):
        return self.build_channels((density,), targets=('j',))[0][0]

    def build_K(self, density):
        return self.build_channels((density,), targets=('k',))[1][0]

    def build_K_pair(self, alpha, beta):
        result = self.build_channels((alpha, beta), targets=('k', 'k'))[1]
        return result[0], result[1]

    def build_jk_uhf(self, alpha, beta):
        self.requirements(('j', 'k', 'k'))
        self._density(alpha); self._density(beta)
        j, k, _ = self.build_channels((alpha+beta, alpha, beta), targets=('j', 'k', 'k'))
        return j[0], k[0], k[1]

    def build_g_rhf(self, density, alpha_hf=1.):
        from .symmetry_molecular_hf import ScreenedMolecularJK
        return ScreenedMolecularJK.build_g_rhf(self, density, alpha_hf)

    def uses_runtime_schwarz_threshold(self):
        return False

    def set_schwarz_threshold(self, threshold):
        pass

    def set_incremental(self, enabled):
        pass

    def incremental_active(self):
        return False
