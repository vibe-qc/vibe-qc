"""Experimental R5.1 same-source MP2 kernels on retained canonical factors.

Moller and Plesset, Phys. Rev. 46, 618 (1934); periodic normalization as in
McClain et al., JCTC 13, 1209 (2017). No refitting or denominator regularization.
The finite-source Gram convention is (pq|rs) = L[pq] L[sr]^H.
"""
from dataclasses import dataclass
import json
import numbers

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


class CanonicalFactorAdapter:
    """Borrow the *executed* canonical metric frame, including its null space.

    Both current three-dimensional builders retain positive modes only. Signed
    metrics are refused, never converted to absolute values. No change of rank,
    whitening, gauge, auxiliary basis or normalization occurs in this adapter.
    """
    def __init__(self, operator):
        from ...periodic_fitted_support import ChiFittedOperator
        from ...periodic_fit_source import factor_digest
        if not isinstance(operator, ChiFittedOperator):
            raise TypeError('correlation needs an explicit ChiFittedOperator')
        d = operator.descriptor
        if d.source not in ('classic', 'sr') or d.auxiliary_frame != 'canonical' or d.factor_space != 'ao-pair':
            raise ValueError('correlation requires the retained canonical AO metric frame')
        if operator.kernel != 'coulomb':
            raise NotImplementedError('R5.1 requires full-Coulomb HF')
        if factor_digest(operator) != d.factor_sha256:
            raise ValueError('source factor identity mismatch')
        if tuple((p, a.shape) for p, a in sorted(operator.items())) != d.factor_shapes:
            raise ValueError('source factor shape metadata mismatch')
        domains = json.loads(d.executed_domains_json)
        records = domains['q_records']
        self.operator = operator
        self.source_identity = operator.identity
        self.domain_record = d.executed_domains_json
        self.metric_records = tuple(records)
        self.nk = len(d.kpoints_cart)
        if set(operator) != {(i, j) for i in range(self.nk) for j in range(self.nk)}:
            raise ValueError('incomplete character pair factors')
        if not np.allclose(operator._weights, np.full(self.nk, 1/self.nk), atol=1e-14, rtol=0):
            raise ValueError('complete characters require uniform weights')
        lattice = np.asarray(records[0]['lattice'])
        k = np.asarray(d.kpoints_cart) @ lattice / (2*np.pi)
        if not np.any(np.max(abs(k-np.rint(k)),axis=1) < 1e-10):
            raise NotImplementedError("shifted characters need a separate admission contract")
        self.k_fractional = k
        self.conserved = np.empty((self.nk, self.nk, self.nk), dtype=int)
        for i in range(self.nk):
            for j in range(self.nk):
                for a in range(self.nk):
                    delta = k[i]+k[j]-k[a]-k
                    match = np.flatnonzero(np.max(abs(delta-np.rint(delta)), axis=1) < 1e-10)
                    if len(match) != 1:
                        raise ValueError('characters do not form a unique complete momentum group')
                    self.conserved[i,j,a] = match[0]
        # Match records to q by value, not by incidental dictionary order.
        covered = set()
        self.pair_groups = {}
        ranks = []
        for q, pairs in d.q_grouping:
            matches = [r for r in records if np.max(abs(np.asarray(r['q'])-q)) < 1e-12]
            if len(matches) != 1:
                raise ValueError('missing or ambiguous metric record')
            record = matches[0]
            eigenvalues = np.asarray(record['metric_eigenvalues'], dtype=float)
            keep = np.asarray(record['metric_keep'])
            # SR records name their executed decomposition (#179): the
            # Cholesky-first fit keeps full rank, the eigen fallback the cut.
            # (vibeqc.aux_basis._rsgdf_metric_keep, inlined: restart proofs
            # forbid native imports here.)
            decomposition = record.get('metric_decomposition', 'eigen')
            if decomposition not in ('cholesky', 'eigen'):
                raise ValueError('invalid retained metric decomposition metadata')
            expected = (np.ones(eigenvalues.shape, dtype=bool) if decomposition == 'cholesky'
                        else eigenvalues > d.metric_threshold)
            if (eigenvalues.ndim != 1 or keep.dtype != bool or keep.shape != eigenvalues.shape
                    or not np.isfinite(eigenvalues).all() or not keep.any()
                    or np.any(eigenvalues[keep] <= 0)
                    or not np.array_equal(keep, expected)):
                raise ValueError('invalid retained metric rank/sign metadata')
            if 'rank' in record and record['rank'] != int(keep.sum()):
                raise ValueError('metric rank metadata mismatch')
            for pair in pairs:
                pair = tuple(pair)
                if pair in covered or pair not in operator:
                    raise ValueError('invalid momentum grouping')
                delta = k[pair[1]]-k[pair[0]]-np.asarray(q) @ lattice/(2*np.pi)
                if np.max(abs(delta-np.rint(delta))) > 1e-10:
                    raise ValueError('factor momentum label mismatch')
                if operator[pair].shape[0] != len(eigenvalues):
                    raise ValueError('canonical metric space dimension mismatch')
                self.pair_groups[pair] = tuple(q)
                covered.add(pair)
            ranks.append(int(keep.sum()))
        if covered != set(operator):
            raise ValueError('incomplete metric momentum grouping')
        self.retained_ranks = tuple(ranks)

    def validate(self):
        from ...periodic_fit_source import factor_digest
        if self.operator.identity != self.source_identity or factor_digest(self.operator) != self.operator.descriptor.factor_sha256:
            raise ValueError('correlation source mutated')

    def integral(self, left, right, ci, ca, cb, cj, batch_size):
        """Return (ia|jb), using two factors in the SAME q metric space.

        right is (kb,kj), not (kj,kb). This avoids assuming unrelated q and
        minus-q eigenvectors have matching phases, including non-TRIM q.
        """
        if self.pair_groups[left] != self.pair_groups[right]:
            raise ValueError("integral factors must share one momentum metric space")
        l, r = self.operator[left], self.operator[right]
        if l.shape[0] != r.shape[0]:
            raise ValueError('incompatible metric spaces')
        result = np.zeros((ca.shape[1], cb.shape[1]), complex)
        for start in range(0, l.shape[0], batch_size):
            x = np.einsum('p,Ppq,qa->Pa', ci.conj(), l[start:start+batch_size], ca)
            y = np.einsum('pb,Ppq,q->Pb', cb.conj(), r[start:start+batch_size], cj)
            result += x.T @ y.conj()
        return result


@dataclass(frozen=True)
class CanonicalMP2Result:
    e_hf_per_cell: float
    e_aa_per_cell: float
    e_bb_per_cell: float
    e_ab_per_cell: float
    active_spaces: tuple
    source_identity: str
    reference_identity: str
    retained_metric_ranks: tuple
    method: str
    source_descriptor_json: str
    scf_acceptance_json: str
    normalization: str = 'energy per primitive cell; complete uniform characters; Nk^-3'
    method_maturity: str = 'experimental'
    quantitative_status: str = 'not-qualified'
    public_admission: str = 'not-requested: developer evaluation'
    restart_mode: str = 'fresh'
    resumed_work_units: int = 0
    completed_work_units: int = 0
    total_work_units: int = 0
    restart_identity: str = ''
    ecp_json: str = '{}'
    electron_counts_json: str = '{}'

    @property
    def backend(self):
        return 'chi-same-source-' + self.method.lower()

    @property
    def e_corr_per_cell(self):
        return self.e_aa_per_cell + self.e_bb_per_cell + self.e_ab_per_cell

    @property
    def energy(self):
        return self.e_hf_per_cell + self.e_corr_per_cell

    @property
    def e_total_per_cell(self):
        return self.energy


def _spin_state(hf, method, tolerance):
    s = np.asarray(hf.overlap)
    channels = []
    for spin in ('alpha', 'beta'):
        suffix = '' if method == 'RHF' else '_'+spin
        c = np.asarray(getattr(hf, 'mo_coeffs'+suffix))
        e = np.asarray(getattr(hf, 'mo_energies'+suffix))
        f = np.asarray(getattr(hf, 'fock'+suffix))
        d = np.asarray(getattr(hf, 'density'+suffix)) / (2 if method == 'RHF' else 1)
        occ = np.asarray(getattr(hf, 'occupations'+suffix)) / (2 if method == 'RHF' else 1)
        if (c.ndim != 3 or e.shape != (len(c), c.shape[2]) or occ.shape != e.shape
                or s.shape != (len(c), c.shape[1], c.shape[1]) or f.shape != s.shape or d.shape != s.shape
                or any(not np.isfinite(x).all() for x in (s,c,e,f,d,occ)) or np.iscomplexobj(e)):
            raise ValueError('invalid canonical spin state dimensions or values')
        if not np.all((abs(occ) < 1e-12) | (abs(occ-1) < 1e-12)):
            raise ValueError('fractional occupations are unsupported')
        counts = np.count_nonzero(occ > .5, axis=1)
        if np.any(counts != counts[0]):
            raise ValueError('metallic character-dependent occupied counts are unsupported')
        n = int(counts[0])
        expected = np.zeros_like(occ); expected[:,:n] = 1
        if not np.allclose(occ, expected, atol=1e-12, rtol=0):
            raise ValueError('canonical Aufbau occupations required')
        if n and n < e.shape[1] and np.min(e[:,n:]) <= np.max(e[:,:n]):
            raise ValueError('metallic or degenerate occupied/virtual gap')
        errors = [np.max(abs(c.conj().transpose(0,2,1) @ s @ c - np.eye(c.shape[2]))),
                  np.max(abs(f @ c - (s @ c)*e[:,None,:])),
                  np.max(abs(c[:,:,:n] @ c[:,:,:n].conj().transpose(0,2,1)-d))]
        if max(errors) > tolerance:
            raise ValueError('reference is not a matched stationary canonical state')
        channels.append((c,e,n))
    return channels


def _double_work_counts(adapter, channels, frozen):
    return tuple(int(adapter.nk**3 * (channels[s][2]-frozen[s]) *
                     (channels[t][2]-frozen[t])) for s, t in ((0, 0), (1, 1), (0, 1)))


def _double_terms(adapter, channels, frozen, batch_size, denominator_tolerance, start=0):
    """Yield one virtual-block contribution in a stable occupied-pair order.

    The integer cursor identifies the exact lexicographic prefix in
    (spin sector, ki, kj, ka, occupied i, occupied j). Skipping that prefix
    does not transform factors or evaluate previously accumulated terms.
    """
    nk, offset = adapter.nk, 0
    for component, (spin, other) in enumerate(((0, 0), (1, 1), (0, 1))):
        c, e, n = channels[spin]; z, h, m = channels[other]
        ni, nj = n-frozen[spin], m-frozen[other]
        count = nk**3 * ni * nj
        for index in range(max(0, start-offset), count):
            block, j = divmod(index, nj)
            block, i = divmod(block, ni)
            block, ka = divmod(block, nk)
            ki, kj = divmod(block, nk)
            i += frozen[spin]; j += frozen[other]
            kb = adapter.conserved[ki, kj, ka]
            denominator = e[ki, i]+h[kj, j]-e[ka, n:, None]-h[kb, None, m:]
            if not np.isfinite(denominator).all() or np.any(denominator >= -denominator_tolerance):
                raise ValueError('MP2 denominator is nonnegative or too small; no regularization applied')
            g = adapter.integral((ki, ka), (kb, kj), c[ki, :, i], c[ka, :, n:],
                                 z[kb, :, m:], z[kj, :, j], batch_size)
            if spin == other:
                exchange = adapter.integral((ki, kb), (ka, kj), c[ki, :, i], c[kb, :, n:],
                                            c[ka, :, n:], c[kj, :, j], batch_size).T
                contribution = .25 * np.sum(abs(g-exchange)**2 / denominator).real
            else:
                contribution = np.sum(abs(g)**2 / denominator).real
            yield component, float(contribution)
        offset += count


def _contract(adapter, channels, frozen, batch_size, denominator_tolerance):
    """Ordered same-spin pairs carry 1/4; alpha-beta pairs carry 1.

    Factors are primitive-cell normalized; supercell integrals divide by Nk.
    The resulting supercell energy divides by Nk once more: overall Nk**-3.
    Only a single occupied pair's virtual block is allocated at a time.
    """
    energies = [0., 0., 0.]
    for component, contribution in _double_terms(adapter, channels, frozen, batch_size, denominator_tolerance):
        energies[component] += contribution
    return tuple(float(energy / adapter.nk**3) for energy in energies)


_MP2_RESTART_SCHEMA = 'vibeqc.chi.canonical-mp2-restart/v1'
_MP2_WORK_ORDER = 'singles-spin-then-doubles-sector-ki-kj-ka-i-j/v1'


def _prefix_counts(completed, counts):
    """Exact coverage of each component in a contiguous work prefix."""
    result = []
    for count in counts:
        done = min(completed, count)
        result.append(done)
        completed -= done
    return np.asarray(result, dtype=np.int64)


def _persistent_contract(reference, adapter, channels, frozen, batch_size,
                         denominator_tolerance, *, checkpoint_path, resume_from,
                         checkpoint_interval, method, conventions=(), singles_term=None):
    """Resume a checked deterministic prefix without retaining per-pair data.

    Numeric accumulators remain unnormalized until the final doubles sum.
    ROHF singles have their own two components and already carry Nk^-1.
    The retained AO factors and canonical frames still dominate memory; this
    work ledger adds only five scalar accumulators and coverage counters.
    A valid checksum establishes storage integrity, not scientific acceptance.
    """
    from ...output.checkpoint import read_restart_state, write_restart_state
    from .checkpoint import portable_reference_identity
    from .reference import _digest
    interval = _positive_integer(checkpoint_interval, 'checkpoint_interval')
    if adapter.operator.descriptor.source != 'classic':
        raise NotImplementedError('persistent MP2 restart supports classic only; SR remains gated')
    nsingles = 2 if singles_term is not None else 0
    counts = ((1, 1) if nsingles else ()) + _double_work_counts(adapter, channels, frozen)
    total = sum(counts)
    binding = dict(method=method, reference_identity=portable_reference_identity(reference),
        source_identity=adapter.source_identity, retained_metric_ranks=list(adapter.retained_ranks),
        canonical_channels_identity=_digest(channels), frozen=list(map(int, frozen)),
        auxiliary_batch_size=batch_size, denominator_tolerance=float(denominator_tolerance),
        conventions=list(conventions), work_order=_MP2_WORK_ORDER,
        momentum_identity=_digest(adapter.conserved), component_work_units=list(counts))
    identity = _digest(binding)
    completed, resumed = 0, 0
    accumulators = np.zeros(len(counts), dtype=np.float64)
    status = 'unfinished-correlation'
    if resume_from is not None:
        metadata, arrays = read_restart_state(resume_from)
        if (metadata.get('schema') != _MP2_RESTART_SCHEMA or
                set(metadata) != {'schema', 'binding', 'identity', 'status', 'completed_work_units', 'total_work_units'}):
            raise ValueError('unsupported or malformed MP2 restart schema; no implicit migration')
        if metadata['binding'] != binding or metadata['identity'] != identity:
            raise ValueError('MP2 restart source, canonical space, or numerical controls are incompatible')
        completed = metadata['completed_work_units']
        if (isinstance(completed, bool) or not isinstance(completed, int)
                or not 0 <= completed <= total or type(metadata['total_work_units']) is not int
                or metadata['total_work_units'] != total):
            raise ValueError('invalid MP2 restart work coverage')
        status = metadata['status']
        if status != ('completed-result' if completed == total else 'unfinished-correlation'):
            raise ValueError('MP2 restart completion status disagrees with work coverage')
        if set(arrays) != {'accumulators', 'component_completed'}:
            raise ValueError('invalid MP2 restart accumulator arrays')
        a, coverage = np.asarray(arrays['accumulators']), np.asarray(arrays['component_completed'])
        expected = _prefix_counts(completed, counts)
        if (a.shape != (len(counts),) or a.dtype != np.dtype('float64') or not np.isfinite(a).all()
                or coverage.dtype != np.dtype('int64') or coverage.shape != expected.shape
                or not np.array_equal(coverage, expected) or np.any(a[expected == 0] != 0)
                or np.any(a > 0)):
            raise ValueError('MP2 restart accumulators or component coverage are invalid')
        accumulators = a.copy()
        resumed = completed
    path = checkpoint_path if checkpoint_path is not None else resume_from

    def publish():
        adapter.validate()
        reference.validate(reference.request_identity, reference.method)
        write_restart_state(path, dict(schema=_MP2_RESTART_SCHEMA, binding=binding,
            identity=identity, status='completed-result' if completed == total else 'unfinished-correlation',
            completed_work_units=completed, total_work_units=total),
            dict(accumulators=accumulators, component_completed=_prefix_counts(completed, counts)))

    if resume_from is None:
        publish()
    for spin in range(completed, nsingles):
        accumulators[spin] += singles_term(spin)
        completed += 1
        if completed % interval == 0 and completed != total:
            publish()
    for component, contribution in _double_terms(adapter, channels, frozen, batch_size,
            denominator_tolerance, start=max(0, completed-nsingles)):
        accumulators[component+nsingles] += contribution
        completed += 1
        if completed % interval == 0 and completed != total:
            publish()
    if completed != total:
        raise RuntimeError('MP2 work iterator failed its exact coverage contract')
    if status != 'completed-result' or checkpoint_path is not None:
        publish()
    mode = ('restored-completed' if status == 'completed-result' else
            'resumed-correlation' if resume_from is not None else 'fresh')
    info = dict(restart_mode=mode, resumed_work_units=resumed,
                completed_work_units=completed, total_work_units=total, restart_identity=identity)
    doubles = tuple(float(x/adapter.nk**3) for x in accumulators[nsingles:])
    return doubles, tuple(float(x) for x in accumulators[:nsingles]), info


def prepare_canonical_reference(reference, *, frozen_core=0, auxiliary_batch_size=64,
                              denominator_tolerance=1e-12):
    """Developer acceptance entry point on a sealed explicit-source HF state.

    Public callers use correlation.run_chi_mp2 for capability admission.
    Frozen core is an explicit number of lowest occupied orbitals per character
    and per spin (or a pair of counts for UHF); no atomic-core guess is made.
    """
    from .reference import ChiHFReference
    if not isinstance(reference, ChiHFReference):
        raise TypeError('same-source MP2 requires a sealed ChiHFReference')
    hf = reference.validate(reference.request_identity, reference.method)
    if reference.method not in ('RHF','UHF'):
        raise NotImplementedError('only RHF MP2 and UHF UMP2 are implemented')
    assembly = getattr(hf, 'exact_exchange_assembly', None)
    if (getattr(assembly, 'c_full', None) != 1.0 or getattr(assembly, 'c_sr', None) != 0.0
            or getattr(hf.aiccm2026dev_b, 'backend', None) != 'ri'
            or getattr(hf, 'returned_operator', 'iterated') != 'iterated'):
        raise NotImplementedError('same-source correlation requires uncorrected full-exchange RI HF')
    if getattr(hf, 'smearing_temperature', 0) != 0:
        raise ValueError('smearing is unsupported')
    report = hf.chi_scf_acceptance
    for key, limit in (('energy_residual',1e-11), ('energy_consistency_error',1e-11),
                       ('orbital_residual',1e-9), ('density_refill_residual',1e-9)):
        value = report.get(key, np.inf)
        if not np.isfinite(value) or not 0 <= value <= limit:
            raise ValueError('correlation needs achieved tight stationary HF convergence')
    from .ecp import validate_ecp_reference
    validate_ecp_reference(hf)
    if getattr(hf, 'executed_chi_xc', None) is not None:
        raise NotImplementedError('KS and hybrid correlation is not implemented')
    adapter = CanonicalFactorAdapter(hf.fitted_operator)
    source = hf.executed_fit_source
    record = json.loads(source.executed_domains_json)
    if record.get('fitted_operator_identity') != adapter.source_identity:
        raise ValueError('HF and correlation source identities differ')
    if not np.allclose(hf.kpoints_cart, adapter.operator.descriptor.kpoints_cart, atol=1e-14, rtol=0):
        raise ValueError('HF and correlation characters differ')
    batch = _positive_integer(auxiliary_batch_size, 'auxiliary_batch_size')
    if not np.isfinite(denominator_tolerance) or denominator_tolerance <= 0:
        raise ValueError('denominator_tolerance must be finite and positive')
    if not np.allclose(hf.kpoint_weights, np.full(adapter.nk,1/adapter.nk), atol=1e-14, rtol=0):
        raise ValueError('HF and correlation normalization differ')
    channels = _spin_state(hf, reference.method, 1e-9)
    frozen = (frozen_core, frozen_core) if isinstance(frozen_core, numbers.Integral) else tuple(frozen_core)
    if len(frozen) != 2 or any(isinstance(v,bool) or not isinstance(v,numbers.Integral) or v < 0 or v > channels[s][2] for s,v in enumerate(frozen)):
        raise ValueError('invalid frozen-core counts')
    if reference.method == 'RHF' and frozen[0] != frozen[1]:
        raise ValueError('restricted frozen cores must agree')
    for spin, (_, energies, occupied) in enumerate(channels):
        core = frozen[spin]
        if core and np.any(np.diff(energies[:,:occupied],axis=1) < 0):
            raise ValueError('frozen-core selection requires occupied orbitals ordered by energy')
        if 0 < core < occupied and np.any(energies[:,core]-energies[:,core-1] <= denominator_tolerance):
            raise ValueError('frozen-core boundary cuts a degenerate occupied subspace')
    return hf, adapter, channels, frozen, batch, source, report


def evaluate_same_source_mp2(reference, *, frozen_core=0, auxiliary_batch_size=64,
                              denominator_tolerance=1e-12, checkpoint_path=None,
                              resume_from=None, checkpoint_interval=1):
    """Developer acceptance entry point on a sealed explicit-source HF state.

    Public callers use correlation.run_chi_mp2 for capability admission.
    Frozen core is an explicit number of lowest occupied orbitals per character
    and per spin (or a pair of counts for UHF); no atomic-core guess is made.
    Classic sources may persist a deterministic work prefix to
    ``checkpoint_path`` or resume one from ``resume_from``. When only the
    latter is provided, subsequent checkpoints replace that same archive.
    """
    hf, adapter, channels, frozen, batch, source, report = prepare_canonical_reference(
        reference, frozen_core=frozen_core, auxiliary_batch_size=auxiliary_batch_size,
        denominator_tolerance=denominator_tolerance)
    restart = {}
    if checkpoint_path is not None or resume_from is not None:
        energies, _, restart = _persistent_contract(reference, adapter, channels, frozen,
            batch, denominator_tolerance, checkpoint_path=checkpoint_path, resume_from=resume_from,
            checkpoint_interval=checkpoint_interval, method='MP2' if reference.method == 'RHF' else 'UMP2')
    else:
        energies = _contract(adapter, channels, frozen, batch, denominator_tolerance)
        total = sum(_double_work_counts(adapter, channels, frozen))
        restart = dict(completed_work_units=total, total_work_units=total)
    adapter.validate()
    reference.validate(reference.request_identity, reference.method)
    from .ecp import electron_accounting
    counts = electron_accounting(hf, frozen, tuple(c[2] for c in channels))
    return CanonicalMP2Result(float(hf.energy), *energies,
        tuple((int(frozen[s]), n-int(frozen[s]), c.shape[-1]-n) for s,(c,e,n) in enumerate(channels)),
        adapter.source_identity, reference.state_identity, adapter.retained_ranks,
        'MP2' if reference.method == 'RHF' else 'UMP2',
        json.dumps(source.as_dict(),sort_keys=True), json.dumps(dict(report),sort_keys=True),
        ecp_json=json.dumps(getattr(hf, 'chi_ecp', None) or {}, sort_keys=True),
        electron_counts_json=json.dumps(counts, sort_keys=True), **restart)
