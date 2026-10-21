"""Complete entry norms for immutable cyclic occupied representations.

Nejad et al., doi:10.1063/5.0290816, Sec. II, Eqs. (3), (9), (10)
and the translation identity after (13): C[R,T]=h[R-T]. With the negative
Fourier transform c(k)=sum_R exp(-ikR)h[R], the home-row symbols are
G_XY(k)=c_X(k).H S(k)c_Y(k) and P_X(k)=c_X(k)c_X(k).H S(k).
The latter is C C.T S (Riplinger/Neese, doi:10.1063/1.4773581, II B1),
including admitted metric asymmetry; it must NOT use S(k).H.

Every distinct displacement block is recovered with the negative transform
divided by N. Only then is the maximum entry taken. A symbol maximum or a
home-cell sample would change the norm. Frozen and active symbols are never
merged. The call-local cache retains scalar measurements and at most one
overlap plus two occupied symbol tensors, bound to immutable content
identities. It never retains complete coefficients or AO projectors.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from math import prod

import numpy as np

from .translation_operators import CyclicAOOperator

_context = ContextVar('occupied_audit_context', default=None)


@contextmanager
def occupied_audit_context():
    """Reuse bounded symbols and sixteen scalar defects in one preparation."""
    if _context.get() is not None:
        yield _context.get()
        return
    state = dict(values={}, hits=0, evaluations=0, boundary_replays=0, source_binding=None,
                 arrays={}, array_bytes=0, array_limit=0, array_hits=0, array_evaluations=0)
    token = _context.set(state)
    try:
        yield state
    finally:
        state['arrays'].clear()
        state['array_bytes'] = 0
        _context.reset(token)


def measured(key, evaluate):
    state = _context.get()
    if state is not None:
        key = (state['source_binding'], *key)
    if state is not None and key in state['values']:
        state['hits'] += 1
        return state['values'][key]
    result = float(evaluate())
    if not np.isfinite(result):
        raise FloatingPointError('non-finite cyclic occupied audit')
    if state is not None:
        state['evaluations'] += 1
        if len(state['values']) < 16:
            state['values'][key] = result
    return result


def bind_source(identity, array_limit):
    state = _context.get()
    if state is not None and state['source_binding'] != identity:
        state['values'].clear()
        state['arrays'].clear()
        state['array_bytes'] = 0
        state['array_limit'] = array_limit
        state['source_binding'] = identity


def _array(key, evaluate):
    state = _context.get()
    if state is not None and key in state['arrays']:
        state['array_hits'] += 1
        return state['arrays'][key]
    result = evaluate()
    if state is not None:
        state['array_evaluations'] += 1
    if not np.isfinite(result).all():
        raise FloatingPointError('non-finite cyclic audit symbols')
    if (state is not None and len(state['arrays']) < 3
            and state['array_bytes']+result.nbytes <= state['array_limit']):
        result.flags.writeable = False
        state['arrays'][key] = result
        state['array_bytes'] += result.nbytes
    return result


def cyclic_inputs(operator, *columns):
    from .translation_storage import TranslationOccupiedColumns
    return (type(operator) is CyclicAOOperator and all(
        type(c) is TranslationOccupiedColumns and c.mesh == operator.mesh
        and c.shape[0] == operator.shape[0] for c in columns))


def column_key(columns):
    return (columns.mesh, columns.home.shape, columns.record()['home_sha256'])


def column_symbols(columns):
    n = prod(columns.mesh)
    b, m = columns.home.shape[0]//n, columns.home.shape[1]
    return _array(('columns', column_key(columns)), lambda: np.fft.fftn(
        columns.home.reshape(*columns.mesh, b, m), axes=(0, 1, 2)).reshape(n, b, m))


def operator_symbols(operator):
    from .translation_preparation import symbols
    return _array(('operator', operator.identity), lambda: symbols(operator))


def displacement_blocks(symbols, mesh):
    return np.fft.fftn(symbols.reshape(*mesh, *symbols.shape[1:]),
                       axes=(0, 1, 2)).reshape(symbols.shape)/prod(mesh)


def _at_boundary(error, tolerance, operator, *columns):
    if tolerance is None:
        return False
    scale = max(1., float(np.max(abs(operator.blocks))))
    for c in columns:
        scale *= max(1., float(np.max(abs(c.home), initial=0.)))
    guard = 128*np.finfo(float).eps*operator.shape[0]*scale
    near = abs(error-tolerance) <= guard
    if near and _context.get() is not None:
        _context.get()['boundary_replays'] += 1
    return near


def metric_defect(left, operator, right, diagonal, tolerance=None):
    def evaluate():
        if not left.shape[1] or not right.shape[1]:
            return 0.
        l = column_symbols(left)
        r = l if left is right else column_symbols(right)
        g = l.conj().transpose(0, 2, 1) @ operator_symbols(operator) @ r
        blocks = displacement_blocks(g, operator.mesh).real
        if diagonal:
            blocks[0] -= np.eye(blocks.shape[1])
        error = float(np.max(abs(blocks), initial=0.))
        if _at_boundary(error, tolerance, operator, left, right):
            from .translation_storage import _streamed_orthogonality_error
            return _streamed_orthogonality_error(left, operator, None if diagonal else right)
        return error
    return measured(('metric', operator.identity, column_key(left), column_key(right), diagonal, tolerance), evaluate)


def projector_defect(columns, operator, target, tolerance=None):
    def evaluate():
        c = column_symbols(columns)
        p = (c @ c.conj().transpose(0, 2, 1)) @ operator_symbols(operator)
        error = float(np.max(abs(displacement_blocks(p, operator.mesh).real-target.blocks), initial=0.))
        if _at_boundary(error, tolerance, operator, columns, columns):
            from .translation_storage import occupied_projector_rows
            b = operator.blocks.shape[1]
            error = 0.
            for start in range(0, operator.shape[0], b):
                rows = np.arange(start, start+b)
                error = max(error, float(np.max(abs(occupied_projector_rows(columns, operator, rows)
                    -target.block(rows, np.arange(operator.shape[0]))))))
        return error
    return measured(('projector', operator.identity, column_key(columns), target.identity, tolerance), evaluate)


def audit_record(operator, active, frozen):
    state = _context.get()
    n, b = prod(operator.mesh), operator.blocks.shape[1]
    m = active.home.shape[1]+frozen.home.shape[1]
    return dict(formulation='complete-cyclic-displacement-blocks', displacement_blocks=n,
                metric_blocks=3*n, projector_blocks=2*n,
                boundary_replays=0 if state is None else state['boundary_replays'],
                intermediate_payload_reservation_bytes=16*8*n*(b*b+b*m+m*m),
                scalar_cache_entries=0 if state is None else len(state['values']),
                scalar_cache_hits=0 if state is None else state['hits'],
                scalar_evaluations=0 if state is None else state['evaluations'],
                scalar_cache_limit=16,
                symbol_cache_hits=0 if state is None else state['array_hits'],
                symbol_transforms=0 if state is None else state['array_evaluations'],
                cached_array_bytes=0 if state is None else state['array_bytes'],
                cache_limit_bytes=0 if state is None else state['array_limit'])


def audit_execution():
    """Final counters before the preparation releases its cache."""
    state = _context.get()
    if state is None:
        return None
    return dict(scalar_evaluations=state['evaluations'], scalar_cache_hits=state['hits'],
                symbol_transforms=state['array_evaluations'], symbol_cache_hits=state['array_hits'],
                cached_array_bytes=state['array_bytes'], cache_limit_bytes=state['array_limit'],
                boundary_replays=state['boundary_replays'], lifetime='preparation-only')
