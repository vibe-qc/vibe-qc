"""Internal experimental translation-native restricted CCSD residual evaluator.

No solver/production method is exposed here. All residual equations execute in
``dlpno_ccsd.cpp`` through the same contraction templates as molecular PNO CCSD.
Riplinger/Neese, doi:10.1063/1.4773581, Sec. II C Eqs. (3)-(18),(30): each
contracted virtual index remains in its amplitude's space; singles products
are evaluated using complete AO tails. Only the operand access is translated.

For each home target, occupied sums visit every cell and active band once,
without orbit weights. The scalar energy alone uses unordered-pair orbit
multiplicity times (2-delta_ij)/N; home singles contribute once per band.
The finite translation/reversal action and half-cell sewing are inherited
from TranslationPairSpaces. Frozen columns occur in PAO recipes, never in
active excitation sums. Strong-pair complete inventories only: screened/weak
pair CC corrections, iterations, DIIS, triples and independent spins are not
implemented by this internal entry point.
"""
from collections import OrderedDict
from math import prod

import numpy as np

from .translation_builder import TranslationPNOBuild, _record
from .translation_source import admit_translation_source
from .translation_transport import prepare_translation_transport, _transport_input_record, _byte_budget
from .translation_storage import ProjectedPairCoefficients, occupied_columns
from .translation_operators import matrix_apply
from .translation_factors import TranslationHomeFactors
from .translation_bloch_factors import TranslationBlochFactors
from .source import source_identity


class _Operands:
    """Solve-local bounded operand and AO-frame reuse; no placed-pair pool."""
    def __init__(self, build, factors, singles, doubles, transport):
        self.spaces = build.spaces
        self.factors = factors
        self.singles, self.doubles = singles, doubles
        self.transport = transport
        self.cache, self.frames = OrderedDict(), OrderedDict()
        self.calls = self.hits = self.reconstructions = 0
        self.peak_cache_bytes = 0

    def clear(self):
        self.cache.clear()
        self.frames.clear()

    def basis(self, i, j):
        key = (i, j)
        if key in self.frames:
            self.frames.move_to_end(key)
            return self.frames[key]
        if len(self.frames) == 2:
            self.frames.popitem(last=False)
        value = self.spaces.coefficients(i, j)
        self.frames[key] = value
        self.reconstructions += 1
        return value

    def occupied(self, i):
        return occupied_columns(self.spaces._occupied, [i])

    def single_ao(self, i):
        return self.basis(i, i) @ self.singles[i % self.spaces._n_bands][:, None]

    def mo(self, left, right):
        value = np.asarray(self.factors.mo_transform(left, right))
        if np.iscomplexobj(value) or not np.isfinite(value).all():
            raise ValueError('translation CCSD requires admitted real factor operands')
        return value

    def __call__(self, name, args):
        self.calls += 1
        key = (name, *args)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        if len(self.cache) == 4:
            self.cache.popitem(last=False)
        value = np.ascontiguousarray(self.compute(name, args), dtype=float)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError('invalid translation CCSD operand')
        value.setflags(write=False)
        self.cache[key] = value
        self.peak_cache_bytes = max(self.peak_cache_bytes,
            sum(x.nbytes for x in self.cache.values())+sum(x.nbytes for x in self.frames.values()))
        return value

    def compute(self, name, args):
        s = self.spaces
        no = s._n_occupied
        i = args[0]
        if name == 'single':
            return self.singles[i % s._n_bands][:, None]
        if name == 'w':
            t = self.single_ao(i)
            out = np.empty((self.factors.n_aux, no))
            for m in range(no):
                out[:, m] = self.mo(self.occupied(m), t)[:, 0, 0]
            return out
        j = args[1]
        if name == 't2':
            return s.orient_tensor(i, j, self.doubles)
        if name == 'boo':
            return self.mo(self.occupied(i), self.occupied(j))[:, 0, :]
        if name == 'ft1':
            return self.occupied(i).T @ matrix_apply(s._fock, self.single_ao(j))
        if name == 'u':
            return self.mo(self.single_ao(i), self.basis(j, args[2]))[:, 0, :]
        if name == 'overlap':
            a, b = s.resolve(i, j), s.resolve(*args[2:])
            shift = tuple((y-x) % n for x, y, n in zip(a.translation, b.translation, s.mesh))
            return self.transport._payload.cross_link(a.row, b.row, int(np.ravel_multi_index(shift, s.mesh)))
        V = self.basis(i, j)
        if name == 'bvv':
            return self.mo(V, V).reshape(self.factors.n_aux, -1)
        if name == 'bov':
            return self.mo(self.occupied(args[2]), V)[:, 0, :]
        if name == 'fvv':
            return V.T @ matrix_apply(s._fock, V)
        if name == 'fov_row':
            return self.occupied(args[2]).T @ matrix_apply(s._fock, V)
        if name == 'fov':
            FV = matrix_apply(s._fock, V)
            out = np.empty((no, V.shape[1]))
            for m in range(no):
                out[m] = self.occupied(m)[:, 0] @ FV
            return out
        if name == 'k':
            Bi = self.mo(self.occupied(i), V)[:, 0, :]
            Bj = self.mo(self.occupied(j), V)[:, 0, :]
            return Bi.T @ Bj
        if name == 'mixed':
            m, k, a, b, dynamic = args[2:]
            weight = self.mo(self.occupied(m), self.single_ao(k) if dynamic else self.occupied(k))[:, 0, 0]
            blocks = self.mo(V, self.basis(a, b))
            return np.einsum('P,Pab->ab', weight, blocks, optimize=False)
        if name == 'x3':
            k = args[2]
            other = j if k == i else i
            Bf = self.mo(self.occupied(other), V)[:, 0, :]
            mixed = self.mo(self.basis(k, k), V)
            return np.einsum('Pf,Pae->fea', Bf, mixed, optimize=False).reshape(V.shape[1]**2, -1)
        raise ValueError('unknown translation CCSD operand')


def evaluate_translation_ccsd(source, build, singles, doubles, *, transport=None,
                              max_workspace_bytes=256*1024**2, max_transport_bytes=256*1024**2,
                              terms=False):
    """Complete internal restricted residual/energy, with no convergence claim.

    Only compact recipes and sealed real home/Bloch factors are supported.
    ``max_workspace_bytes`` covers conservative CC numerical payload, operand
    cache/reconstruction/conversion work and returned residual/term arrays.
    Admission/transport and factor-provider work retain their separate budgets;
    pre-existing source/build/amplitudes, external BLAS scratch and container/allocator
    overhead are excluded. This is not MP2's memory bound.
    """
    from .._vibeqc_core import _translation_ccsd_workspace, _translation_ccsd_evaluate
    if not isinstance(build, TranslationPNOBuild):
        raise TypeError('translation CCSD requires a complete TranslationPNOBuild')
    s = build.spaces
    if type(s.pair_coefficients) is not ProjectedPairCoefficients:
        raise ValueError('translation CCSD requires compact complete PAO/PNO recipes')
    if type(source.factors) not in (TranslationHomeFactors, TranslationBlochFactors):
        raise ValueError('translation CCSD requires sealed translation factors')
    if not isinstance(terms, bool):
        raise TypeError("terms must be boolean")
    cap = _byte_budget(max_workspace_bytes, 'max_workspace_bytes')
    no, ao, rank = s._n_occupied, s._occupied.shape[0], max(build.pno_ranks)
    aux, npair, bands = source.factors.n_aux, len(build.pno_ranks), s._n_bands
    native_bytes = int(_translation_ccsd_workspace(no, aux, rank, npair, bool(terms)))
    # Four operand cache entries, two AO frames, live consumers/replacements,
    # contraction products, C-order conversion and immutable output copies.
    python_bytes = 4096+8*(64*aux*(rank*rank+no+rank)+64*ao*rank+32*rank**3
                           +16*npair*rank*rank+16*no*rank+8*no*no)
    required = native_bytes+python_bytes
    if required > cap:
        raise MemoryError(f'translation CCSD workspace requires {required} bytes, exceeds max_workspace_bytes')
    inputs = admit_translation_source(source, s.mesh, s._occupied,
        frozen_occupied=s._frozen_occupied, tolerance=s.tolerance)
    if (inputs.source_identity != build.source_identity
        or inputs.factors.identity != build.provenance['factor_identity']
        or source_identity(_record(inputs.provenance)) != source_identity(_record(build.provenance['source_snapshot']))):
        raise ValueError('translation CCSD source does not match pair build')
    if transport is None:
        transport = prepare_translation_transport(build, backend='bounded', max_bytes=max_transport_bytes)
    if (transport.provenance['backend'] != 'bounded'
        or transport.input_identity != source_identity(_transport_input_record(build))):
        raise ValueError('translation CCSD transport does not match bounded pair build')
    if transport.required_bytes > _byte_budget(max_transport_bytes, 'max_transport_bytes'):
        raise MemoryError('translation CCSD transport exceeds max_transport_bytes')
    doubles = s.validate_tensors(doubles, name='CCSD doubles')
    singles = tuple(np.asarray(v) for v in singles)
    if len(singles) != bands:
        raise ValueError('CCSD singles require one vector per home active band')
    for i, v in enumerate(singles):
        if (v.shape != (s.pair_ranks[s.resolve(i, i).row],) or np.iscomplexobj(v)
            or not np.isfinite(v).all()):
            raise ValueError('CCSD singles shape/real values')
    singles = tuple(np.array(v, dtype=float, copy=True) for v in singles)
    operands = _Operands(build, source.factors, singles, doubles, transport)
    try:
        result = _translation_ccsd_evaluate(transport._payload,s.occupied_fock,aux,rank,
            operands,native_bytes,terms)
        result['provenance'] = dict(kind='internal-translation-restricted-ccsd-residual-v1',
            source_identity=build.source_identity, build_identity=build.identity,
            transport_identity=transport.identity, same_source=build.same_source,
            complete_strong_pairs=True, iterations=False, scientific_acceptance=False,
            n_representative_singles=bands, n_representative_pairs=npair,
            native_workspace_bytes=native_bytes, python_workspace_bytes=python_bytes,
            required_bytes=required, maximum_bytes=cap, cache_policy='four-operands-two-frames-solve-local',
            operand_calls=operands.calls, operand_cache_hits=operands.hits,
            pair_reconstructions=operands.reconstructions, peak_cache_bytes=operands.peak_cache_bytes,
            cache_bytes_after_evaluation=0)
        result['raw_correlation_energy_per_cell'] = result['energy']
        result['citations'] = ['riplinger_dlpno_2013', 'deprince_sherrill_df_ccsd_2013',
                               'stanton_gauss_watts_bartlett_ccsd_1991', 'nejad_periodic_dlpno_pbc_2025']
        result['pno_mp2_correction_per_cell'] = build.pno_correction_per_cell
        return result
    finally:
        operands.clear()
