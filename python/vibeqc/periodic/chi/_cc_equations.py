"""Complex spin-orbital CCSD algebra, independent of source construction.

Stanton et al., JCP 94, 4334 (1991), DOI 10.1063/1.460620, Eqs. 1-13.
Amplitudes t1[a,i], t2[a,b,i,j] multiply excitation operators; they are
not conjugated in residuals or energy. Integrals g[p,q,r,s]=<pq||rs>
are conjugate symmetric between bra and ket pairs. The virtual ladder
streams its first index. No CCM producer or real-orbital convention is used.
"""
import numpy as np

def _ccsd_energy_so(ts, td, fs, spinints, nocc_so):
    """Spin-orbital CCSD correlation energy (Stanton 1991). ts[a,i], td[a,b,i,j]."""
    o = slice(0, nocc_so)
    v = slice(nocc_so, spinints.shape[0])
    e = np.einsum("ia,ai->", fs[o, v], ts)
    e += 0.25 * np.einsum("ijab,abij->", spinints[o, o, v, v], td)
    e += 0.5 * np.einsum("ijab,ai,bj->", spinints[o, o, v, v], ts, ts)
    return e


def residual(ts, td, fs, spinints, nocc_so):
    """Projected spin-orbital CCSD residual Ω1[a,i], Ω2[a,b,i,j] (Ω = 0 at convergence).

    ``Ω = numerator - D*t``. Off-diagonal Fock blocks remain explicit so
    the equations can be checked against a general determinant Hamiltonian.
    Production chi calls supply canonical orbital energies.

    Reference: Stanton, Gauss, Watts & Bartlett, J. Chem. Phys. 94, 4334 (1991)
    (the "Programmer's Guide" T1/T2 equations, eqs 1-2 and the F/W intermediates).
    """
    nso = spinints.shape[0]
    o = slice(0, nocc_so)
    v = slice(nocc_so, nso)
    eps = np.diag(fs)
    fov = fs[o, v]
    fvv = fs[v, v] - np.diag(np.diag(fs[v, v]))          # off-diagonal f_ab
    foo = fs[o, o] - np.diag(np.diag(fs[o, o]))          # off-diagonal f_ij

    ts_t = td + 0.5 * (np.einsum("ai,bj->abij", ts, ts) - np.einsum("bi,aj->abij", ts, ts))
    ta = td + (np.einsum("ai,bj->abij", ts, ts) - np.einsum("bi,aj->abij", ts, ts))

    fae = (
        fvv
        - 0.5 * np.einsum("me,am->ae", fov, ts)
        + np.einsum("fm,mafe->ae", ts, spinints[o, v, v, v])
        - 0.5 * np.einsum("afmn,mnef->ae", ts_t, spinints[o, o, v, v])
    )
    fmi = (
        foo
        + 0.5 * np.einsum("me,ei->mi", fov, ts)
        + np.einsum("en,mnie->mi", ts, spinints[o, o, o, v])
        + 0.5 * np.einsum("efin,mnef->mi", ts_t, spinints[o, o, v, v])
    )
    fme = fov + np.einsum("fn,mnef->me", ts, spinints[o, o, v, v])

    wmnij = (
        spinints[o, o, o, o]
        + np.einsum("ej,mnie->mnij", ts, spinints[o, o, o, v])
        - np.einsum("ei,mnje->mnij", ts, spinints[o, o, o, v])
        + 0.25 * np.einsum("efij,mnef->mnij", ta, spinints[o, o, v, v])
    )
    # Eq. (7), streamed on the external a index: no vvvv allocation.
    def virtual_ladder(a):
        x = slice(nocc_so+a, nocc_so+a+1)
        return (spinints[x, v, v, v]
            - np.einsum("bm,amef->abef", ts, spinints[x, o, v, v])
            + np.einsum("am,bmef->abef", ts[a:a+1], spinints[v, o, v, v])
            + 0.25 * np.einsum("abmn,mnef->abef", ta[a:a+1], spinints[o, o, v, v]))

    wmbej = (
        spinints[o, v, v, o]
        + np.einsum("fj,mbef->mbej", ts, spinints[o, v, v, v])
        - np.einsum("bn,mnej->mbej", ts, spinints[o, o, v, o])
        - np.einsum("fbjn,mnef->mbej",
                    (0.5 * td + np.einsum("fj,bn->fbjn", ts, ts)),
                    spinints[o, o, v, v])
    )

    # T1 numerator (Stanton eq 1) — the bare f_ai drive is nonzero off-canonical.
    ts_num = (
        fs[v, o]
        + np.einsum("ei,ae->ai", ts, fae)
        - np.einsum("am,mi->ai", ts, fmi)
        + np.einsum("aeim,me->ai", td, fme)
        - np.einsum("fn,naif->ai", ts, spinints[o, v, o, v])
        - 0.5 * np.einsum("efim,maef->ai", td, spinints[o, v, v, v])
        - 0.5 * np.einsum("aemn,nmei->ai", td, spinints[o, o, v, o])
    )
    Dai = (eps[o, None] - eps[None, v]).T                # [a,i] = εi − εa
    omega1 = ts_num - Dai * ts

    # T2 numerator (Stanton eq 2).
    t2 = np.array(spinints[v, v, o, o], dtype=complex, copy=True)
    tmp = fae - 0.5 * np.einsum("bm,me->be", ts, fme)
    t2 += np.einsum("aeij,be->abij", td, tmp)
    t2 -= np.einsum("beij,ae->abij", td, tmp)            # P(ab)
    tmp = fmi + 0.5 * np.einsum("ej,me->mj", ts, fme)
    t2 -= np.einsum("abim,mj->abij", td, tmp)
    t2 += np.einsum("abjm,mi->abij", td, tmp)            # P(ij)
    t2 += 0.5 * np.einsum("abmn,mnij->abij", ta, wmnij)
    for a in range(len(ts)):
        t2[a:a+1] += 0.5 * np.einsum("efij,abef->abij", ta, virtual_ladder(a))
    ring = np.einsum("aeim,mbej->abij", td, wmbej) - np.einsum(
        "ei,am,mbej->abij", ts, ts, spinints[o, v, v, o])
    t2 += ring
    t2 -= ring.transpose(0, 1, 3, 2)                     # P(ij)
    t2 -= ring.transpose(1, 0, 2, 3)                     # P(ab)
    t2 += ring.transpose(1, 0, 3, 2)                     # P(ij)P(ab)
    dt = np.einsum("ei,abej->abij", ts, spinints[v, v, v, o])
    t2 += dt - dt.transpose(0, 1, 3, 2)                  # P(ij)
    dm = np.einsum("am,mbij->abij", ts, spinints[o, v, o, o])
    t2 -= dm - dm.transpose(1, 0, 2, 3)                  # P(ab)
    Dabij = (
        eps[o, None, None, None] + eps[None, o, None, None]
        - eps[None, None, v, None] - eps[None, None, None, v]
    ).transpose(2, 3, 0, 1)                              # [a,b,i,j]
    omega2 = t2 - Dabij * td
    return omega1, omega2



def triples(t1, t2, eps, integrals, *, denominator_tolerance=1e-12,
            restart_state=None, progress_callback=None, checkpoint_interval=1):
    """Canonical spin-orbital Hermitian (T), Raghavachari 1989 Eq. 14.

    For unique ordered triples: sum C* (C + D) / Delta, where
    C = <abc_ijk| (V T2)_connected |0>, D = <abc_ijk| V T1 |0>.
    This is the explicitly specified complex Hermitian extension of the real
    canonical RHF/UHF expression, not a general-reference or ROHF definition.
    All aaa, aab, abb and bbb sectors occur in the unique spin-orbital sum.
    Full antisymmetrization divided by four accounts for each seed's two
    already antisymmetric index pairs. Unique triples remove the 1/36.
    No six-index tensor is allocated and no denominator is regularized.

    Numerical progress covers the exact lexicographic prefix of all unique
    occupied/virtual triples, including forbidden zero-contribution units.
    Its rolling digest verifies coverage; the persistent caller must also bind
    source, amplitudes, orbital energies and controls before using this state.
    A partial accumulation is only emitted through the callback, never returned
    as a completed correction.
    """
    from itertools import combinations, permutations
    from hashlib import sha256
    from math import comb
    if (isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, (int, np.integer))
            or checkpoint_interval < 1):
        raise ValueError('triples checkpoint_interval must be a positive integer')
    eps = np.asarray(eps)
    if (eps.ndim != 1 or np.iscomplexobj(eps) or not np.isfinite(eps).all()
            or not np.isfinite(denominator_tolerance) or denominator_tolerance <= 0):
        raise ValueError('triples requires finite real orbital energies and positive denominator tolerance')
    nv, no = t1.shape
    if (eps.shape != (nv+no,) or t2.shape != (nv,nv,no,no)
            or not np.isfinite(t1).all() or not np.isfinite(t2).all()):
        raise ValueError('triples amplitude/energy dimensions disagree')
    total = comb(no, 3)*comb(nv, 3) if min(no,nv) >= 3 else 0
    completed, evaluated, energy = 0, 0, 0j
    expected_digest = None
    if restart_state is not None:
        if set(restart_state) != {'completed_work', 'evaluated_work', 'total_work', 'coverage_sha256', 'energy'}:
            raise ValueError('triples checkpoint coverage fields are incompatible')
        for name in ('completed_work', 'evaluated_work', 'total_work'):
            value = restart_state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('triples checkpoint coverage counters are invalid')
        completed, evaluated = restart_state['completed_work'], restart_state['evaluated_work']
        if restart_state['total_work'] != total or not 0 <= evaluated <= completed <= total:
            raise ValueError('triples checkpoint work coverage is incompatible')
        if np.shape(restart_state['energy']) != (1,) or not np.isfinite(restart_state['energy']).all():
            raise ValueError('triples checkpoint accumulated energy is invalid')
        expected_digest = restart_state['coverage_sha256']
        energy = complex(restart_state['energy'][0])
        if completed == 0 and (evaluated != 0 or energy != 0j):
            raise ValueError('empty triples checkpoint has nonzero accumulated work')
    coverage = sha256(b'chi-spin-orbital-triples-lexicographic/v1')
    if completed == 0 and expected_digest is not None and expected_digest != coverage.hexdigest():
        raise ValueError('triples checkpoint prefix coverage does not match')
    def publish(done, used):
        if progress_callback is not None:
            progress_callback(dict(completed_work=done, evaluated_work=used,
                total_work=total, coverage_sha256=coverage.hexdigest(),
                energy=np.array([energy], dtype=complex)))
    if total == 0:
        publish(0, 0)
        return 0.0
    o, v = slice(0,no), slice(no,no+nv)
    vvvo = integrals[v,v,v,o]
    ovoo = integrals[o,v,o,o]
    vvoo = integrals[v,v,o,o]
    perms = [(p, (-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3)))
             for p in permutations(range(3))]
    allowed = getattr(integrals, 'allowed_excitation', None)
    ordinal, used = 0, 0
    for occ in combinations(range(no),3):
        for vir in combinations(range(nv),3):
            ordinal += 1
            admissible = allowed is None or allowed(occ, tuple(no+a for a in vir))
            used += int(admissible)
            coverage.update(np.asarray((*occ, *vir, int(admissible)), dtype='<i8').tobytes())
            if ordinal <= completed:
                if ordinal == completed and (expected_digest != coverage.hexdigest() or used != evaluated):
                    raise ValueError('triples checkpoint prefix coverage does not match')
                continue
            if not admissible:
                if ordinal % checkpoint_interval == 0 or ordinal == total:
                    publish(ordinal, used)
                continue
            denominator = sum(eps[i] for i in occ)-sum(eps[no+a] for a in vir)
            if not np.isfinite(denominator) or denominator >= -denominator_tolerance:
                raise ValueError('triples denominator is nonnegative or too small')
            connected, disconnected = 0j, 0j
            for po, so in perms:
                i,j,k = (occ[x] for x in po)
                for pv, sv in perms:
                    a,b,c = (vir[x] for x in pv)
                    connected += so*sv*(np.dot(t2[a,:,i,j],vvvo[b,c,:,k])
                                        -np.dot(t2[a,b,i,:],ovoo[:,c,j,k]))/4
                    disconnected += so*sv*t1[a,i]*vvoo[b,c,j,k]/4
            energy += connected.conjugate()*(connected+disconnected)/denominator
            if not np.isfinite(energy):
                raise ValueError('nonfinite partial triples energy')
            if ordinal % checkpoint_interval == 0 or ordinal == total:
                publish(ordinal, used)
    if not np.isfinite(energy) or abs(energy.imag)>1e-9:
        raise ValueError('non-real or nonfinite triples energy')
    return float(energy.real)
