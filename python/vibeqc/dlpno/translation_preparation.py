"""Cyclic symmetric Lowdin selection, with a scoped numerical-payload ledger.

The unitary transform U[R,k]=exp(ikR)/sqrt(N) is Nejad et al.,
doi:10.1063/5.0290816, Sec. II, Eqs. (2)-(10). For normalized AO seeds E,
B(k)=Q(k)E, G(k)=B(k).H S(k)B(k). The full seed Gram is unitarily
similar to diag_k G(k); its symmetric inverse root has symbols G(k)^-1/2.
Home columns are N^-1 sum_k exp(ikR) B(k)G(k)^-1/2, not N^-1/2.
All eigenvalues enter conditioning. No eigenspace or coefficient is screened.
"""
from math import prod
from numbers import Integral

import numpy as np

from .translation_operators import CyclicAOOperator
from .translation_storage import TranslationOccupiedColumns

DEFAULT_SELECTION_BYTES = 64 * 1024**2


def selection_workspace(mesh, local, bands, maximum):
    """Preflight conservative simultaneous *controlled array payload*.

    Reserves symbols/FFT input, output and copies (including source-projector
    realification), home buffers and snapshots, exhaustive AO row checks,
    per-character eigensolver inputs/outputs and NumPy iterator buffers.
    LAPACK/FFT/BLAS internal workspace, Python metadata and allocator overhead
    are outside this numerical-payload cap. The bounded symbol/scalar cache
    expires when the immutable preparation ends.
    """
    if isinstance(maximum, (bool, np.bool_)) or not isinstance(maximum, Integral) or maximum <= 0:
        raise ValueError('max_selection_bytes must be a positive integer')
    n, b, m = prod(mesh), int(local), int(bands)
    charges = dict(symbols_transforms_and_copies=16*16*n*b*b,
                   homes_snapshots_and_reconstruction=8*32*n*b*m,
                   exhaustive_audit_tiles=8*12*n*b*b,
                   cyclic_audit_intermediates=16*8*n*(b*b+b*m+m*m),
                   audit_symbol_cache=16*n*(b*b+b*m),
                   scalar_audit_cache=16*8,
                   character_validation=64*n*n,
                   eigensolver_arrays=16*24*(2*b)**2+8*16*b,
                   indices_and_phases=8*16*n*(b+m+3),
                   numpy_iterator_buffers=16*8*np.getbufsize())
    required = sum(charges.values())
    if required > maximum:
        raise MemoryError(f'occupied selection requires {required} accounted bytes; max_selection_bytes={maximum}')
    return dict(kind='cyclic-occupied-preparation-payload-v2', max_selection_bytes=int(maximum),
                accounted_peak_bytes=required, charges=charges, persistent_cache_bytes=0,
                preparation_symbol_cache_bytes=charges['audit_symbol_cache'],
                source_recipe_reservation_bytes=32*n*m*m,
                home_output_bytes=8*n*b*m, maximum_eigensolver_order=2*b,
                scope='compact-source-occupied-audit-and-selection; excludes caller/source arrays, '
                      'native library workspace, PAO/PNO/transport/solver, Python metadata and RSS')


def symbols(operator):
    return (np.fft.ifftn(operator.blocks.reshape(*operator.mesh, *operator.blocks.shape[1:]),
                         axes=(0, 1, 2))*prod(operator.mesh)).reshape(operator.blocks.shape)


def _inverse_characters(mesh):
    return [np.ravel_multi_index(tuple((-np.asarray(k)) % mesh), mesh) for k in np.ndindex(mesh)]


def cyclic_projected_frame(projector, overlap, rank, tolerance, seed_lindep, label):
    n, b = prod(projector.mesh), projector.blocks.shape[1]
    m = rank//n
    if m == 0:
        return TranslationOccupiedColumns.from_home(np.empty((n*b, 0)), projector.mesh), (), ()
    q, s = symbols(projector), symbols(overlap)
    inverse = _inverse_characters(projector.mesh)
    norms = np.sqrt(np.diag(overlap.blocks[0]))

    def gram(k, seeds):
        # Retain Q.H S Q, including finite-precision non-idempotence.
        # Self-inverse characters of a real cyclic operator have real
        # symbols. Use their entire real block, without splitting modes.
        qk, sk = (q[k].real, s[k].real) if k == inverse[k] else (q[k], s[k])
        columns = qk[:, seeds]/norms[seeds]
        g = columns.conj().T @ sk @ columns
        return .5*(g+g.conj().T), columns

    selected, minima = [], []
    for _ in range(m):
        best, score = None, -np.inf
        for candidate in range(b):
            if candidate in selected:
                continue
            low, high = np.inf, -np.inf
            for k in range(n):
                values = np.linalg.eigvalsh(gram(k, [*selected, candidate])[0])
                if not np.isfinite(values).all():
                    raise FloatingPointError('non-finite projected-AO seed Gram matrix')
                low, high = min(low, float(values[0])), max(high, float(values[-1]))
            if high <= 0 or low <= seed_lindep*high:
                continue
            if best is None or low > score+tolerance:
                best, score = candidate, low
        if best is None:
            raise ValueError(f'{label} projector has no admitted full-rank translated AO seed frame; '
                             'the bounded common-seed selector cannot construct this gauge')
        selected.append(best)
        minima.append(score)
    transformed = np.empty((n, b, m), dtype=complex)
    low, high = np.inf, -np.inf
    for k in range(n):
        g, columns = gram(k, selected)
        values, vectors = np.linalg.eigh(g)
        low, high = min(low, float(values[0])), max(high, float(values[-1]))
        # A matrix function retains complete degeneracies and fixes the same
        # symmetric gauge regardless of eigenvector phases or ordering.
        root = (vectors/np.sqrt(values)) @ vectors.conj().T
        transformed[k] = columns @ root
    if not np.isfinite(transformed).all() or high <= 0 or low <= seed_lindep*high:
        raise ValueError(f'{label} selected AO seed frame is numerically rank deficient')
    # Do not make independent real/imaginary orbitals or split TRIM modes.
    # Reconstruct all characters, checking reality in the AO-entry norm.
    home = np.fft.ifftn(transformed.reshape(*projector.mesh, b, m), axes=(0, 1, 2)).reshape(n*b, m)
    if np.max(abs(home.imag), initial=0.) > tolerance:
        raise ValueError(f'{label} selected frame has no real translation gauge')
    return TranslationOccupiedColumns.from_home(home.real, projector.mesh), tuple(selected), tuple(minima)


def real_occupied_projector(coefficients, overlaps, phase, mesh, label, tolerance, *, kfrac=None,
                            audit='auto'):
    """Exact character decomposition of the original [Re Z, Im Z] audit.

    A_k=[C_k,conj(C_-k)]/sqrt(2) is unitarily equivalent to that real
    span. At self-inverse characters use [Re C, Im C] explicitly. Thus the
    union of the 2m Gram spectra is precisely the original 2Nm spectrum.
    Require m unit and m zero eigenvalues per character, without a cutoff.
    The normalized span projector is compared with the original *complex*
    source projector over every real-space displacement and AO entry.
    """
    n, b, m = coefficients.shape
    if audit not in ('auto', 'exhaustive'):
        raise ValueError('occupied audit must be auto or exhaustive')
    if not m:
        return CyclicAOOperator(np.zeros((n, b, b)), mesh), dict(
            projector_error=0., orthogonality=0., imaginary=0., translation_error=0.,
            metric_audit='zero-rank-cyclic-blocks', complete_column_replays=0,
            displacement_blocks=n, cached_array_bytes=0)
    # Private immutable operands bind all recipes and audits to this call.
    # These snapshots are charged by symbols_transforms_and_copies.
    def snapshot(value):
        a = np.ascontiguousarray(value, dtype=complex)
        return np.frombuffer(a.tobytes(), dtype=complex).reshape(a.shape)
    coefficients, overlaps, phase = map(snapshot, (coefficients, overlaps, phase))
    if kfrac is None:
        inverse = _inverse_characters(mesh)
    else:
        characters = np.rint(np.asarray(kfrac)*mesh).astype(int) % mesh
        lookup = {tuple(k): i for i, k in enumerate(characters)}
        inverse = [lookup[tuple((-k) % mesh)] for k in characters]
    expected = np.empty((n, b, b), complex)
    recipes = np.empty((n, 2*m, m), complex)

    def spans(k):
        neg, c = inverse[k], coefficients[k]
        sc = overlaps[k] @ c
        span = (np.column_stack((c.real, c.imag)) if k == neg else
                np.column_stack((c, coefficients[neg].conj()))/np.sqrt(2))
        metric_span = (np.column_stack((sc.real, sc.imag)) if k == neg else
                       np.column_stack((sc, (overlaps[neg] @ coefficients[neg]).conj()))/np.sqrt(2))
        return span, metric_span

    for k in range(n):
        span, metric_span = spans(k)
        g = span.conj().T @ metric_span
        values, vectors = np.linalg.eigh(.5*(g+g.conj().T))
        if max(np.max(abs(values[:-m])), np.max(abs(values[-m:]-1))) > tolerance:
            raise ValueError(f'{label} occupied projector does not admit a real frame')
        recipes[k] = vectors[:, -m:]/np.sqrt(values[-m:])
        expected[k] = coefficients[k] @ (overlaps[k] @ coefficients[k]).conj().T

    # Reuse compact coefficient-space recipes, never cache folded frames.
    # A conjugate pair supplies both real columns together; a TRIM supplies
    # its entire real span. No degeneracy is split by a numerical threshold.
    representatives = [k for k, neg in enumerate(inverse) if k <= neg]
    def realify(k):
        span, metric_span = spans(k)
        frame, metric_frame = span @ recipes[k], metric_span @ recipes[k]
        z = (phase[k, :, None, None]*frame[None, :, :]).reshape(n*b, m)/np.sqrt(n)
        sz = (phase[k, :, None, None]*metric_frame[None, :, :]).reshape(n*b, m)/np.sqrt(n)
        if k == inverse[k]:
            return z.real, sz.real
        return np.column_stack((z.real, z.imag))*np.sqrt(2), np.column_stack((sz.real, sz.imag))*np.sqrt(2)

    source_blocks = np.einsum('kd,kab->dab', phase.conj(), expected, optimize=False)/n
    sblocks = np.einsum('kd,kab->dab', phase.conj(), overlaps, optimize=False)/n
    S = CyclicAOOperator(sblocks.real, mesh)
    # Establish the actual complete character table and ordering, not a
    # caller's covariance declaration. Noncanonical tables retain replay.
    cells = np.asarray(list(np.ndindex(mesh)))
    fractions = cells/np.asarray(mesh) if kfrac is None else np.asarray(kfrac)
    characters = np.rint(fractions*mesh).astype(int) % mesh
    canonical = characters/np.asarray(mesh)
    verified = (n == prod(mesh) and len(set(map(tuple, characters))) == n
                and np.max(abs(fractions-canonical), initial=0.) <= 8*np.finfo(float).eps
                and np.array_equal(phase, np.exp(2j*np.pi*(fractions @ cells.T))))
    if audit == 'auto' and verified:
        result, record = _cyclic_span_audit(spans, recipes, representatives, inverse,
                                          characters, phase, S, source_blocks)
        # At a floating-point boundary replay the original arithmetic. This
        # guard is NOT an acceptance tolerance; every final comparison still
        # uses the caller's original tolerance, including tiny tolerances.
        guard = (128*np.finfo(float).eps*n*b*max(1., float(np.max(abs(overlaps))))
                 * max(1., float(np.max(abs(coefficients))))**2)
        defect = max(record['projector_error'], record['orthogonality'], record['translation_error'])
        if abs(defect-tolerance) > guard:
            if defect > tolerance:
                raise ValueError(f'{label} real frame changes the occupied metric projector')
            return result, record
    orthogonality = 0.
    for k in representatives:
        frame, metric_frame = realify(k)
        # Preserve both the original real-span metric and the actual admitted
        # real operator check. Test every cross-character orbital entry.
        metric = S.apply(frame)
        for j in representatives:
            other, _ = realify(j)
            for action in (metric_frame, metric):
                gram = other.T @ action
                if j == k:
                    gram -= np.eye(gram.shape[0])
                orthogonality = max(orthogonality, float(np.max(abs(gram))))
    error = covariance = 0.
    projector = None
    cells = tuple(np.ndindex(mesh))
    for r, cell in enumerate(cells):
        actual = np.zeros((b, n*b))
        span_actual = np.zeros_like(actual)
        for k in representatives:
            frame, metric_frame = realify(k)
            actual += frame[r*b:(r+1)*b] @ S.apply(frame, transpose=True).T
            span_actual += frame[r*b:(r+1)*b] @ metric_frame.T
        if projector is None:
            projector = CyclicAOOperator(actual.reshape(b, n, b).transpose(1, 0, 2), mesh)
        for t, target in enumerate(cells):
            d = np.ravel_multi_index(tuple((np.asarray(target)-cell) % mesh), mesh)
            reference = source_blocks[d]
            error = max(error, float(np.max(abs(span_actual[:, t*b:(t+1)*b]-reference))))
            covariance = max(covariance, float(np.max(abs(actual[:, t*b:(t+1)*b]-projector.blocks[d]))))
    if max(error, covariance, orthogonality) > tolerance:
        raise ValueError(f'{label} real frame changes the occupied metric projector')
    return projector, dict(projector_error=error, orthogonality=orthogonality,
        imaginary=float(np.max(abs(source_blocks.imag))), translation_error=covariance,
        real_span_spectrum='complete-character-blocks', metric_audit='exhaustive-real-column-replay',
        complete_column_replays=len(representatives)**2+(1+n)*len(representatives),
        displacement_blocks=n, cyclic_structure_verified=verified,
        replay_reason=('requested' if audit == 'exhaustive' else
                       'threshold-boundary' if verified else 'unverified-character-table'))


def _cyclic_span_audit(spans, recipes, representatives, inverse, characters, phase, S, source_blocks):
    """Original real-orbital norm and every AO displacement, without replay.

    For a conjugate character pair the real columns are sqrt(2)[Re UF,
    Im UF]. Their metric block is [[Re g, Im g],[-Im g, Re g]], where
    g=F.H M; distinct character pairs are orthogonal. Thus take maxima of
    real and imaginary entries separately, NOT abs(g). At a self-inverse
    character use the complete real block. Check both the original M and
    S_real(k)F, without replacing either by an identity or its adjoint.

    The two projectors have symbols F M.H and F F.H S_real(k), respectively.
    Conjugate-pair partners are reconstructed from the same recipe (not an
    independently chosen degenerate gauge). Inverse transformation visits
    every displacement. This is Nejad Sec. II Eqs. (3),(9),(10), with the
    occupied projector of Riplinger/Neese Sec. II B1, not a new criterion.
    """
    n, b = S.blocks.shape[:2]
    actual_symbols = np.empty((n, b, b), complex)
    span_symbols = np.empty_like(actual_symbols)
    sk = symbols(S)
    orthogonality = 0.
    for k in representatives:
        a, ma = spans(k)
        f, mf = a @ recipes[k], ma @ recipes[k]
        if k == inverse[k]:
            f, mf = f.real, mf.real
        canonical_index = np.ravel_multi_index(tuple(characters[k]), S.mesh)
        s = sk[canonical_index]
        for action in (mf, s @ f):
            g = f.conj().T @ action-np.eye(f.shape[1])
            orthogonality = max(orthogonality, float(np.max(abs(g.real), initial=0.)),
                                float(np.max(abs(g.imag), initial=0.)))
        span_symbols[k] = f @ mf.conj().T
        actual_symbols[k] = (f @ f.conj().T) @ s
        if inverse[k] != k:
            span_symbols[inverse[k]] = span_symbols[k].conj()
            actual_symbols[inverse[k]] = actual_symbols[k].conj()
    span_blocks = np.einsum('kd,kab->dab', phase.conj(), span_symbols, optimize=False)/n
    actual_blocks = np.einsum('kd,kab->dab', phase.conj(), actual_symbols, optimize=False)/n
    error = float(np.max(abs(span_blocks-source_blocks), initial=0.))
    # The original translation defect is identically zero under the proven
    # character law: Q[R+G,T+G]-Q[R,T]=Q[T-R]-Q[T-R]. It is not the
    # imaginary reconstruction norm (which is reported independently).
    covariance = 0.
    return CyclicAOOperator(actual_blocks.real, S.mesh), dict(
        projector_error=error, orthogonality=orthogonality,
        imaginary=float(np.max(abs(source_blocks.imag), initial=0.)), translation_error=covariance,
        real_span_spectrum='complete-character-blocks', metric_audit='verified-real-character-blocks',
        projector_audit='complete-cyclic-displacement-blocks', cyclic_structure_verified=True,
        translation_audit='verified-displacement-permutations',
        reconstruction_imaginary=float(np.max(abs(actual_blocks.imag), initial=0.)),
        complete_column_replays=0, real_character_blocks=len(representatives),
        displacement_blocks=n, intermediate_payload_reservation_bytes=16*8*n*b*b,
        cached_array_bytes=0)
