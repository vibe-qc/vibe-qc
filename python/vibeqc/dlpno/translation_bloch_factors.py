"""Bounded retained-Bloch factors for the real translation engine.

Nejad et al., doi:10.1063/5.0290816, Sec. II, Eqs. (2)-(10), gives the
unitary finite-character convention. With c[k]=sum_R exp(-ikR) C[R]/sqrt(N),

 B[tP,Rmu,Snu] = N^-2 sum_ij exp(iki(R-t)-ikj(S-t)) L[ij,Pmu nu],
 X[q,P,l,r] = N^-1 sum_(j-i=q) c_left[i]^H L[ij,P] c_right[j],
 B_MO[tP,l,r] = sum_q exp(iqt) X[q,P,l,r].

For real orbitals, summing auxiliary cells is Parseval:
 (ia|jb) = N sum_qP X[q,P,i,a] conj(X[q,P,j,b]).
The energy's separate division by N belongs to the shared solver. No metric
inverse, refit, support truncation or orbital overlap is inserted here.
"""
from __future__ import annotations

import cmath
from itertools import product
from numbers import Integral
from types import MappingProxyType

import numpy as np

from .source import source_identity
from .translation_factors import _SealedFactorType
from ..periodic.chi._transform_inputs import checked_mesh_size, integer_array


DEFAULT_FACTOR_BYTES = 64 * 1024**2
FACTOR_TOLERANCE = 1e-10


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _maximum(value):
    if not np.isfinite(value).all():
        raise ValueError('nonfinite retained Coulomb factor transform')
    return float(np.max(np.abs(value), initial=0.))


class TranslationBlochFactors(metaclass=_SealedFactorType):
    """Sealed factor provider borrowing the immutable executed SCF operator.

    The byte cap covers numerical array payload during each serial provider
    call, including returned arrays until return, dtype conversions, phase
    cache, audit tiles, einsum outputs and a NumPy iterator-buffer allowance.
    No factor call crosses a native binding or uses BLAS packing. Source
    storage, reference admission, caller arrays, outputs retained after return,
    Python metadata, allocator overhead and downstream work are outside it.
    Calls on one provider must not overlap. Only one bounded phase block is cached.
    """
    __slots__ = ('_operator', '_mesh', '_k', '_cells', '_groups', '_n_home',
                 '_b', '_identity', '_limit', '_tile', '_cache_limit', '_phase', '_state', '_audit')

    def __init_subclass__(cls, **kwargs):
        raise TypeError('translation factor implementation cannot be subclassed')

    def __setattr__(self, name, value):
        raise AttributeError('translation factors are immutable')

    def __delattr__(self, name):
        raise AttributeError('translation factors are immutable')

    def __init__(self, adapter, mesh, *, max_bytes=DEFAULT_FACTOR_BYTES,
                 auxiliary_tile=16, cache_bytes=1024**2):
        from ..periodic.chi.canonical_mp2 import CanonicalFactorAdapter
        from ..periodic_fitted_support import ChiFittedOperator
        if hasattr(self, '_operator'):
            raise AttributeError('translation factors cannot be reinitialized')
        if type(adapter) is not CanonicalFactorAdapter or type(adapter.operator) is not ChiFittedOperator:
            raise TypeError('streamed factors require the sealed retained canonical adapter')
        adapter.validate()
        mesh = tuple(integer_array(mesh, name='mesh', shape=(3,)).tolist())
        n = checked_mesh_size(mesh)
        if n != adapter.nk:
            raise ValueError('factor mesh differs from retained characters')
        k = tuple(tuple(float(x) for x in row) for row in adapter.k_fractional)
        residues = [tuple(int(round(x*m)) % m for x, m in zip(row, mesh)) for row in k]
        if (len(set(residues)) != n or any(abs(x*m-round(x*m)) > 1e-10
                for row in k for x, m in zip(row, mesh))):
            raise ValueError('factor characters must form the complete unshifted mesh')
        shapes = {value.shape for value in adapter.operator.values()}
        if len(shapes) != 1:
            raise ValueError('factors require a common canonical auxiliary frame')
        a, b, c = next(iter(shapes))
        if not a or not b or b != c:
            raise ValueError('invalid retained factor dimensions')
        limit = _integer(max_bytes, 'max_factor_bytes', 1)
        tile = _integer(auxiliary_tile, 'factor_auxiliary_tile', 1)
        cache = _integer(cache_bytes, 'factor_cache_bytes')
        # A cache ceiling is not a reservation. One immutable phase block may
        # be retained; its construction/copies are charged in every plan.
        for name, value in dict(_operator=adapter.operator, _mesh=mesh, _k=k,
                _cells=tuple(product(*(range(m) for m in mesh))),
                _groups=tuple(tuple(tuple(p) for p in pairs)
                              for _, pairs in adapter.operator.descriptor.q_grouping),
                _n_home=a, _b=b, _limit=limit, _tile=tile, _cache_limit=cache,
                _phase=None,
                _state=dict(cache_hits=0, cache_misses=0, peak_accounted_bytes=0,
                            max_output_bytes=0, max_auxiliary_tile=0, calls=0, busy=False,
                            max_character_tile=0, phase_entries_generated=0,
                            inverse_blocks=0, inverse_entries=0, recovered_entries=0, covariance_entries=0)).items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, '_identity', source_identity(dict(
            kind='translation-retained-bloch-factors-v1', source_identity=adapter.source_identity,
            mesh=mesh, k_fractional=k, normalization='N^-2 home; N Parseval',
            auxiliary_frame='unchanged canonical', tolerance=FACTOR_TOLERANCE)))
        # Preflight before allocating any phase or audit array.
        width = self._plan(0, 0, output_bytes=0, audit=True)
        self._state['audit_character_tile'] = self._state['character_tile']
        self._state['audit_auxiliary_tile'] = width
        object.__setattr__(self, '_audit', MappingProxyType(self._audit_source(width)))

    @property
    def mesh(self): return self._mesh

    @property
    def n_aux(self): return len(self._cells) * self._n_home

    @property
    def n_orb(self): return len(self._cells) * self._b

    @property
    def identity(self): return self._identity

    @property
    def source_identity(self): return self._operator.identity

    @property
    def audit(self): return self._audit

    @property
    def provenance(self):
        n = len(self._cells)
        phase = self._phase
        return dict(kind='translation-retained-bloch-factors-v1', identity=self.identity,
            source_identity=self.source_identity, mesh=self.mesh, n_orb=self.n_orb, n_aux=self.n_aux,
            max_factor_bytes=self._limit, auxiliary_tile_requested=self._tile,
            audit_primitive_ao_tile=1, audit_formulation='complete tiled inverse/forward Fourier',
            cache_policy='one immutable character/displacement block or recompute',
            cache_limit_bytes=self._cache_limit, cache_bytes=0 if phase is None else phase[1].nbytes,
            **{key: self._state[key] for key in ('cache_hits', 'cache_misses',
                'peak_accounted_bytes', 'max_output_bytes', 'max_auxiliary_tile', 'calls',
                'max_character_tile', 'audit_character_tile', 'audit_auxiliary_tile',
                'phase_entries_generated', 'inverse_blocks', 'inverse_entries', 'recovered_entries',
                'covariance_entries')},
            source_storage_bytes=sum(a.nbytes for a in self._operator.values()),
            removed_home_factor_bytes=8*self._n_home*self.n_orb**2,
            retained_home_factor_bytes=0, native_binding_copy_bytes=0,
            max_contraction_imaginary=self._state.get('max_contraction_imaginary', 0.),
            phase_table_bytes=0, removed_phase_table_bytes=32*n*n,
            phase_workspace_bytes=128*n*self._state['max_character_tile'],
            minimum_audit_bytes=self._iterator_bytes()+320*n,
            audit_displacement_tile=self._state['audit_character_tile'],
            audit_complete_home_entries=n*n*self._n_home*self._b**2,
            phase_cache_binding=self.identity,
            iterator_allowance_bytes=self._iterator_bytes(),
            scope='serial provider numerical payload including returned buffers until return; '
                  'excludes source and caller storage, reference admission, previously returned '
                  'outputs, downstream allocations, metadata and allocator overhead',
            audits=dict(self.audit))

    @staticmethod
    def _iterator_bytes():
        # optimize=False einsum avoids BLAS and its opaque packing workspace.
        # Reserve eight complex operand buffers at the current NumPy buffer
        # setting, exceeding the operand count of every einsum/ufunc below.
        return 8 * int(np.getbufsize()) * 16

    def _plan(self, fixed, per_aux, *, output_bytes, audit=False):
        # Linear slabs only, including at generous budgets: d < N for N > 1.
        # Phase charge covers two live blocks and a replacement's array and
        # immutable bytes copy (four two-sign blocks). Audit charge covers the
        # recovery slab, home slab, inverse half-transform, gathered source,
        # reversed block, products, reductions and finite/absolute temporaries.
        n = len(self._cells)
        minimum = self._iterator_bytes()+128*n+fixed+(192*n if audit else per_aux)
        if self._limit < minimum:
            raise MemoryError(f'factor workspace budget {self._limit} bytes is smaller than '
                              f'the required minimum {minimum} bytes')
        for d in range(min(8, max(1, n//2)), 0, -1):
            base = self._iterator_bytes()+128*n*d+fixed
            unit = 192*n*d if audit else per_aux
            width = min(self._tile, self._n_home, (self._limit-base)//unit)
            if width >= 1:
                break
        state = self._state
        state['character_tile'] = d
        state['max_character_tile'] = max(state['max_character_tile'], d)
        state['peak_accounted_bytes'] = max(state['peak_accounted_bytes'], base+width*unit)
        state['max_output_bytes'] = max(state['max_output_bytes'], output_bytes)
        state['max_auxiliary_tile'] = max(state['max_auxiliary_tile'], width)
        # A previous call may have used a larger tile. Evict before allocating.
        if self._phase is not None and self._phase[1].nbytes > 32*n*d:
            object.__setattr__(self, '_phase', None)
        return width

    def _phase_value(self, i, r):
        return cmath.exp(2j*np.pi*sum(x*y for x, y in zip(self._k[i], self._cells[r])))

    def _phases(self, i0, i1, r0, r1):
        """One source/mesh-bound immutable cache entry; never a full table."""
        d = self._state['character_tile']
        if min(i1-i0, r1-r0) > d:
            raise ValueError('phase request exceeds planned character/displacement tile')
        key = (i0, i1, r0, r1)
        state = self._state
        if self._phase is not None and self._phase[0] == key:
            state['cache_hits'] += 1
            return self._phase[1]
        state['cache_misses'] += 1
        object.__setattr__(self, '_phase', None)
        phase = np.empty((2, i1-i0, r1-r0), complex)
        for i in range(i0, i1):
            for r in range(r0, r1):
                phase[0, i-i0, r-r0] = self._phase_value(i, r)
        state['phase_entries_generated'] += (i1-i0)*(r1-r0)
        np.conjugate(phase[0], out=phase[1])
        if phase.nbytes <= self._cache_limit:
            phase = np.frombuffer(phase.tobytes(), dtype=complex).reshape(phase.shape)
            object.__setattr__(self, '_phase', (key, phase))
        return phase

    def _fold(self, r0, r1, s0, s1, start, stop, mu, nu):
        """B[R,S] in a rectangle, with the oracle's i-then-j sum order.

        Nejad et al., doi:10.1063/5.0290816, II Eqs. (2)-(10): writing
        H[R,j,P] = sum_i exp(ik_i R) L[i,j,P,mu,nu]/N gives
        B[R,S,P] = sum_j exp(-ik_j S) H[R,j,P]/N. A complete j slab
        costs N*d*p, not N*N*p. No rank or AO entry is projected away.
        """
        n, d = len(self._cells), self._state['character_tile']
        phase = self._phases(0, n, r0, r1)[0]
        src = np.empty((n, stop-start), complex)
        half = np.empty((r1-r0, n, stop-start), complex)
        for j in range(n):
            for i in range(n):
                src[i] = self._operator[i,j][start:stop,mu,nu]
            np.einsum('ir,ip->rp', phase, src, out=half[:,j], optimize=False)
        half /= n
        del phase, src
        result = np.empty((r1-r0, s1-s0, stop-start), complex)
        for s in range(s0, s1, d):
            end = min(s+d, s1)
            conjugate = self._phases(0, n, s, end)[1]
            np.einsum('js,rjp->rsp', conjugate, half,
                      out=result[:,s-s0:end-s0], optimize=False)
        result /= n
        self._state['inverse_blocks'] += 1
        self._state['inverse_entries'] += result.size
        return result

    def _audit_source(self, width):
        """Full original entry norms with linear character/displacement slabs.

        For each target character tile I, reconstruct every B[R,S] and its
        separately folded reversed AO block B_nu,mu[S,R]. Measure imaginary
        and real symmetry maxima, then A=(Re B+Re B_reverse.T)/2, exactly as
        in the complete-table oracle. Accumulate W[I,S]=sum_R exp(-ik_I R) A
        in original R order and recover L[I,J]=sum_S exp(ik_J S) W. Every
        target I,J,P,mu,nu is compared, including canonical null components.
        Recomputing B for each I avoids storing either quadratic audit array.
        The two full-axis einsum reductions retain the oracle's summation
        order; tiling never substitutes a symbol norm or occupied projection.
        """
        n, d = len(self._cells), self._state['character_tile']
        defects = dict(imaginary=0., symmetry=0., round_trip=0., translation_covariance=0.)
        for start in range(0, self._n_home, width):
            stop = min(start+width, self._n_home)
            for mu in range(self._b):
                for nu in range(self._b):
                    for i0 in range(0, n, d):
                        i1 = min(i0+d, n)
                        recovered = np.zeros((i1-i0, n, stop-start), complex)
                        term = np.empty_like(recovered)
                        for r0 in range(0, n, d):
                            r1 = min(r0+d, n)
                            home = self._fold(r0, r1, 0, n, start, stop, mu, nu)
                            defects['imaginary'] = max(defects['imaginary'], _maximum(home.imag))
                            for s0 in range(0, n, d):
                                s1 = min(s0+d, n)
                                reverse = self._fold(s0, s1, r0, r1, start, stop, nu, mu)
                                view = home[:,s0:s1].real
                                rev = reverse.transpose(1,0,2).real
                                defects['symmetry'] = max(defects['symmetry'], _maximum(view-rev))
                                view += rev
                                view *= .5
                                del reverse, view, rev
                            home.imag.fill(0.)
                            conjugate = self._phases(i0, i1, r0, r1)[1]
                            for r in range(r1-r0):
                                np.multiply(conjugate[:,r,None,None], home[r], out=term)
                                recovered += term
                            del home, conjugate
                        for j0 in range(0, n, d):
                            j1 = min(j0+d, n)
                            phase = self._phases(j0, j1, 0, n)[0]
                            restored = np.empty((i1-i0, j1-j0, stop-start), complex)
                            np.einsum('js,isp->ijp', phase, recovered, out=restored, optimize=False)
                            for i in range(i0, i1):
                                for j in range(j0, j1):
                                    restored[i-i0,j-j0] -= self._operator[i,j][start:stop,mu,nu]
                            defects['round_trip'] = max(defects['round_trip'], _maximum(restored))
                            self._state['recovered_entries'] += restored.size
                            del restored, phase
                        del recovered, term
        # Full actual character action, including wraps and self-inverse cells.
        # A single row suffices to visit all origins/shifts for one character.
        lookup = {cell: r for r,cell in enumerate(self._cells)}
        for i in range(n):
            phase, conjugate = self._phases(i, i+1, 0, n)[:,0]
            for t, shift in enumerate(self._cells):
                for r, cell in enumerate(self._cells):
                    shifted = lookup[tuple((x-y)%m for x,y,m in zip(cell,shift,self.mesh))]
                    error = float(abs(phase[shifted]-phase[r]*conjugate[t]))
                    defects['translation_covariance'] = max(defects['translation_covariance'], error)
                    self._state['covariance_entries'] += 1
        if max(defects.values()) > FACTOR_TOLERANCE:
            raise ValueError(f'retained Coulomb factors fail real symmetric Fourier source audit: {defects}')
        return defects

    def _inputs(self, left, right, *, pair):
        # Requiring ndarray avoids an unbudgeted list-to-array conversion before
        # dimensions can be inspected. Strided and non-double arrays are copied
        # only after the guard, with both conversion buffers charged below.
        for value in (left, right):
            kinds = 'iuf' if pair else 'iufc'
            if (not isinstance(value, np.ndarray) or value.ndim != 2
                    or value.shape[0] != self.n_orb or value.shape[1] == 0
                    or value.dtype.kind not in kinds):
                raise ValueError('factor coefficients must be finite ndarray AO matrices in the admitted scope')
        if pair and left.shape[1] != 2:
            raise ValueError('pair exchange requires two real occupied columns')
        l, r = left.shape[1], right.shape[1]
        output = (8*r*r if pair else
                  (16 if np.iscomplexobj(left) or np.iscomplexobj(right) else 8)*self.n_aux*l*r)
        # Complex output plus real returned copy, contraction products,
        # conversion buffers, Fourier buffers, finiteness/absolute-value
        # buffers, conjugation and half-transform work are all charged.
        fixed = 64*self.n_orb*(l+r) + (48*r*r if pair else 32*self.n_aux*l*r)
        width = self._plan(fixed, 64*l*r+32*l*self._b, output_bytes=output)
        return width

    def _contract(self, left, right, *, pair):
        width = self._inputs(left, right, pair=pair)
        n, b = len(self._cells), self._b
        l, r = left.shape[1], right.shape[1]
        lc, rc = np.empty((n,b,l), complex), np.empty((n,b,r), complex)
        for value, out in ((left,lc), (right,rc)):
            value = np.array(value, dtype=complex, order='C', copy=True)
            if not np.isfinite(value).all():
                raise ValueError('factor coefficients must be finite')
            d = self._state['character_tile']
            for k in range(0, n, d):
                end = min(k+d, n)
                conjugate = self._phases(k, end, 0, n)[1]
                np.einsum('kr,ram->kam', conjugate, value.reshape(n,b,-1),
                          out=out[k:end], optimize=False)
            out /= np.sqrt(n)
        del value
        np.conjugate(lc, out=lc)
        result = np.zeros((r,r) if pair else (n,self._n_home,l,r), complex)
        product_buffer = np.empty((r,r), complex) if pair else None
        for start in range(0, self._n_home, width):
            stop = min(start+width, self._n_home)
            x = np.empty((stop-start,l,r), complex)
            term = np.empty_like(x)
            half = np.empty((stop-start,l,b), complex)
            for group in self._groups:
                x.fill(0.)
                for i,j in group:
                    np.einsum('ml,Pmn->Pln', lc[i], self._operator[i,j][start:stop],
                              out=half, optimize=False)
                    np.einsum('Pln,nr->Plr', half, rc[j], out=term, optimize=False)
                    x += term
                x /= n
                if pair:
                    np.conjugate(x, out=term)
                    np.einsum('Pa,Pb->ab', x[:,0], term[:,1], out=product_buffer, optimize=False)
                    product_buffer *= n
                    result += product_buffer
                else:
                    i,j = group[0]
                    # Two phase rows, never a complete character table.
                    phase_j = self._phases(j, j+1, 0, n)[0,0]
                    phase_i = self._phases(i, i+1, 0, n)[1,0]
                    for t in range(n):
                        np.multiply(x, phase_j[t]*phase_i[t], out=term)
                        result[t,start:stop] += term
                    del phase_i, phase_j
            del x, term, half
        if not np.isfinite(result).all():
            raise FloatingPointError('nonfinite retained factor contraction')
        if pair or not (np.iscomplexobj(left) or np.iscomplexobj(right)):
            # Admission audits the complete factors, not just these outputs.
            # Report the contraction defect as well; do not relax tolerance.
            defect = _maximum(result.imag)
            if defect > FACTOR_TOLERANCE:
                raise ValueError('retained factor contraction is not real within tolerance')
            self._state['max_contraction_imaginary'] = max(
                self._state.get('max_contraction_imaginary', 0.), defect)
            result = result.real.copy()
        return result if pair else result.reshape(self.n_aux,l,r)

    def _call(self, left, right, *, pair):
        if self._state['busy']:
            raise RuntimeError('factor workspace budget requires serial calls on one provider')
        self._state['busy'] = True
        try:
            result = self._contract(left, right, pair=pair)
            self._state['calls'] += 1
            return result
        finally:
            self._state['busy'] = False

    def pair_exchange(self, occupied, virtual):
        return self._call(occupied, virtual, pair=True)

    def mo_transform(self, coefficients_left, coefficients_right=None):
        right = coefficients_left if coefficients_right is None else coefficients_right
        return self._call(coefficients_left, right, pair=False)
