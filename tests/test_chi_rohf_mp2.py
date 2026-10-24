"""Independent finite-Hamiltonian RMP2 evidence, not native acceptance."""
from dataclasses import replace
import itertools
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc.periodic.chi import canonical_rohf_mp2 as rmp
from vibeqc.periodic.chi import canonical_mp2 as mp
from vibeqc.periodic.chi import reference as refs
from vibeqc.periodic_fit_source import RSGDFFitSource, factor_digest
from vibeqc.periodic_fitted_support import ChiFittedOperator


def _ao_eri(op):
    """Four-character AO Hamiltonian with momentum independently enumerated."""
    nk = len(op._weights); nao = op[0, 0].shape[1]
    tensor = np.zeros((nk, nao, nk, nao, nk, nao, nk, nao), complex)
    for i, a, j, b in itertools.product(range(nk), repeat=4):
        if (i+j-a-b) % nk:
            continue
        tensor[i, :, a, :, j, :, b, :] = np.einsum(
            'Ppq,Psr->pqrs', op[i, a], op[b, j].conj())/nk
    return tensor.reshape((nk*nao,)*4)


def _block_diagonal(blocks):
    total_rows = sum(x.shape[0] for x in blocks)
    total_cols = sum(x.shape[1] for x in blocks)
    result = np.zeros((total_rows, total_cols), complex)
    row = col = 0
    for x in blocks:
        result[row:row+x.shape[0], col:col+x.shape[1]] = x
        row += x.shape[0]; col += x.shape[1]
    return result


def reseal(ref):
    state = refs._hf_state_identity(ref.hf_result)
    ref = replace(ref, state_identity=state)
    return replace(ref, seal=refs._digest((ref.owner_pid, ref.request_identity,
        ref.state_identity, ref.method, ref.quantitative_status)))


def fixture(monkeypatch, nk=1, nc=1, no=1, nv=1):
    """Build a genuine stationary high-spin determinant of an AO Hamiltonian.

    Start from explicit ERIs and shared occupied projectors, then construct a
    Hermitian h with the required ROHF gradient zero. No production SCF/J/K
    routines enter this fixture. Physical spin Focks follow from that h and
    those ERIs, with a nonzero equal-and-opposite closed-virtual spin block.
    """
    rng = np.random.default_rng(274+nk)
    nao = nc+no+nv; na = nc+no
    k = np.zeros((nk, 3)); k[:, 0] = np.arange(nk)/nk
    b = rng.normal(size=(2, nao, nao))*.12
    blocks = {0: b+b.transpose(0, 2, 1)}
    for q in range(1, (nk+1)//2):
        t = rng.normal(size=b.shape)*.12+1j*rng.normal(size=b.shape)*.12
        blocks[q], blocks[nk-q] = t, t.conj().transpose(0, 2, 1)
    factors = {(i, a): blocks[(a-i) % nk] for i in range(nk) for a in range(nk)}
    groups = tuple(((q/nk, 0., 0.), tuple(p for p in factors if (p[1]-p[0]) % nk == q)) for q in range(nk))
    records = [dict(q=q, lattice=(2*np.pi*np.eye(3)).tolist(),
                    metric_eigenvalues=[1., 2.], metric_keep=[True, True], rank=2) for q, _ in groups]
    desc = RSGDFFitSource('synthetic-finite', 1, 'classic', 'model', 'orb', 'orb-hash', 'aux', 'aux-hash',
        (), 1e-9, 'canonical', groups, tuple(map(tuple, k)), 10., None,
        tuple((p, a.shape) for p, a in sorted(factors.items())), factor_digest(factors),
        executed_domains_json=json.dumps(dict(q_records=records)))
    op = ChiFittedOperator(factors, desc, np.ones(nk)/nk)
    eri = _ao_eri(op)
    c = np.array([np.linalg.qr(rng.normal(size=(nao, nao))+1j*rng.normal(size=(nao, nao)))[0] for _ in range(nk)])
    da = c[:, :, :na] @ c[:, :, :na].conj().transpose(0, 2, 1)
    db = c[:, :, :nc] @ c[:, :, :nc].conj().transpose(0, 2, 1)
    pa, pb = _block_diagonal(da), _block_diagonal(db)
    j = np.einsum('pqrs,sr->pq', eri, pa+pb)
    ka = np.einsum('prsq,rs->pq', eri, pa)
    kb = np.einsum('prsq,rs->pq', eri, pb)
    hcore, fa, fb = [], [], []
    for ik in range(nk):
        ix = slice(ik*nao, (ik+1)*nao)
        ko = c[ik].conj().T @ (ka-kb)[ix, ix] @ c[ik]
        mean = np.diag(np.arange(nao)*1.7-3.).astype(complex)
        mean[:nc, nc:na] = -.5*ko[:nc, nc:na]
        mean[nc:na, :nc] = mean[:nc, nc:na].conj().T
        mean[nc:na, na:] = .5*ko[nc:na, na:]
        mean[na:, nc:na] = mean[nc:na, na:].conj().T
        mean = c[ik] @ mean @ c[ik].conj().T
        h = mean-j[ix, ix]+.5*(ka+kb)[ix, ix]
        hcore.append(h); fa.append(h+j[ix, ix]-ka[ix, ix]); fb.append(h+j[ix, ix]-kb[ix, ix])
    hcore, fa, fb = map(np.asarray, (hcore, fa, fb))
    energy = float(np.einsum('kpq,kqp->', da, hcore+fa).real/2/nk
                   + np.einsum('kpq,kqp->', db, hcore+fb).real/2/nk)
    occupations = np.zeros((nk, nao)); occupations[:, :nc] = 2; occupations[:, nc:na] = 1
    hf = SimpleNamespace(energy=energy, converged=True, mo_coeffs=c, occupations=occupations,
        mo_energies=np.tile(np.arange(nao)*1.7-3., (nk, 1)), mo_coeffs_alpha=c, mo_coeffs_beta=c,
        occupations_alpha=(occupations > 0).astype(float), occupations_beta=(occupations == 2).astype(float),
        n_closed=nc, n_open=no, n_alpha=na, n_beta=nc, density_alpha=da, density_beta=db,
        fock_alpha=fa, fock_beta=fb, hcore=hcore, overlap=np.tile(np.eye(nao), (nk, 1, 1)),
        kpoint_weights=np.ones(nk)/nk, kpoints_cart=k, smearing_temperature=0.,
        rohf_subspaces=dict(closed=tuple(range(nc)), open=tuple(range(nc, na)), virtual=tuple(range(na, nao))),
        rohf_convention='coulson-roothaan-average-diagonal/v1',
        rohf_canonicalization='closed-open-virtual-average-fock/v1',
        aiccm2026dev_b=dict(electronic_method='ROHF', backend='ri'),
        exact_exchange_assembly=dict(c_full=1., c_sr=0.), process_provenance={'fixture': 1},
        fitted_operator=op, executed_fit_source=replace(desc, executed_domains_json=json.dumps(dict(fitted_operator_identity=op.identity))),
        chi_scf_acceptance=dict(final_rebuild=True, stationary=True, accepted=True,
            generalized_rohf_gradient_norm=0., energy_residual=0., energy_consistency_error=0.,
            density_refill_residual=0., orbital_residual=0.))
    # The state seal accepts dataclasses, not arbitrary namespace metadata.
    from tests.test_chi_canonical_mp2 import Diagnostic, Assembly
    hf.aiccm2026dev_b, hf.exact_exchange_assembly = Diagnostic('ROHF'), Assembly()
    monkeypatch.setattr(refs, 'loaded_process_provenance', lambda: {'fixture': 1})
    ref = refs.ChiHFReference(hf, os.getpid(), 'request', '', 'ROHF')
    return reseal(ref), eri


def _independent_frame(hf, frozen=0):
    """Block spectral projectors, independently assembled in AO coordinates."""
    result = []
    for spin in ('alpha', 'beta'):
        frames, spectra = [], []
        count = getattr(hf, 'n_'+spin)-frozen
        for k, coefficients in enumerate(hf.mo_coeffs):
            c = coefficients.copy()
            if frozen:
                closed = c[:, :hf.n_closed]
                fmean = (hf.fock_alpha[k]+hf.fock_beta[k])/2
                _, u = np.linalg.eigh(closed.conj().T @ fmean @ closed)
                c[:, :hf.n_closed] = closed @ u
                c = c[:, frozen:]
            spin_f = getattr(hf, 'fock_'+spin)[k]
            frame, spectrum = [], []
            for indices in (np.arange(count), np.arange(count, c.shape[1])):
                x = c[:, indices]
                values, vectors = np.linalg.eigh(x.conj().T @ spin_f @ x)
                frame.extend((x @ vectors).T); spectrum.extend(values)
            frames.append(np.asarray(frame).T); spectra.append(spectrum)
        result.append((np.asarray(frames), np.asarray(spectra), count))
    return result


def determinant_oracle(hf, eri):
    """Apply explicit second-quantized H to the reference determinant.

    Gamma only. All determinants of the same N and Ms are generated. The
    second-order coefficient is sum_D |<D|H|0>|^2/(E0(H0)-ED(H0)). No
    production residual, transformed integral, singles or doubles helper.
    """
    assert len(hf.mo_coeffs) == 1
    channels = _independent_frame(hf)
    orbitals = [(s, c[0, :, p], e[0, p]) for s, (c, e, n) in enumerate(channels) for p in range(c.shape[-1])]
    nsp = len(orbitals); nmo = nsp//2
    occupied = tuple(list(range(hf.n_alpha))+list(range(nmo, nmo+hf.n_beta)))
    reference = sum(1 << p for p in occupied)
    h = np.zeros((nsp, nsp), complex); g = np.zeros((nsp,)*4, complex)
    for p, (sp, cp, ep) in enumerate(orbitals):
        for q, (sq, cq, eq) in enumerate(orbitals):
            if sp == sq:
                h[p, q] = cp.conj() @ hf.hcore[0] @ cq
            for r, (sr, cr, er) in enumerate(orbitals):
                for s, (ss, cs, es) in enumerate(orbitals):
                    if sp == sr and sq == ss:
                        g[p, q, r, s] = np.einsum('u,v,w,x,uvwx->', cp.conj(), cr, cq.conj(), cs, eri)

    def apply(mask, operations):
        sign = 1
        for index, create in operations:
            if bool(mask & (1 << index)) == create:
                return None, 0
            sign *= (-1)**((mask & ((1 << index)-1)).bit_count())
            mask ^= 1 << index
        return mask, sign

    amplitudes = {}
    for p, q in itertools.product(range(nsp), repeat=2):
        mask, sign = apply(reference, ((q, False), (p, True)))
        if sign:
            amplitudes[mask] = amplitudes.get(mask, 0)+h[p, q]*sign
    for p, q, r, s in itertools.product(range(nsp), repeat=4):
        mask, sign = apply(reference, ((r, False), (s, False), (q, True), (p, True)))
        if sign:
            amplitudes[mask] = amplitudes.get(mask, 0)+.5*g[p, q, r, s]*sign
    e0 = sum(orbitals[i][2] for i in occupied)
    contributions = {1: 0., 2: 0.}
    for mask, value in amplitudes.items():
        if mask == reference:
            assert value.real == pytest.approx(hf.energy, abs=2e-12)
            continue
        rank = (mask ^ reference).bit_count()//2
        denominator = e0-sum(orbitals[i][2] for i in range(nsp) if mask & (1 << i))
        contributions[rank] += abs(value)**2/denominator
    return contributions[1], contributions[2]


@pytest.mark.parametrize('nc,no,nv', [(1, 1, 1), (1, 2, 1), (1, 0, 2), (0, 1, 2)])
def test_explicit_determinant_perturbation(monkeypatch, nc, no, nv):
    ref, eri = fixture(monkeypatch, nc=nc, no=no, nv=nv)
    result = rmp.evaluate_same_source_rohf_mp2(ref)
    singles, doubles = determinant_oracle(ref.hf_result, eri)
    assert result.e_singles_per_cell == pytest.approx(singles, abs=2e-13)
    assert result.e_doubles_per_cell == pytest.approx(doubles, abs=2e-13)
    assert result.e_corr_per_cell == pytest.approx(singles+doubles, abs=2e-13)
    assert result.e_total_per_cell == pytest.approx(ref.hf_result.energy+singles+doubles)
    assert result.method == 'ROHF-RMP2'
    if nc and no:
        assert singles < -1e-8  # Dropping the non-Brillouin term changes E2.
    else:
        assert abs(singles) < 1e-24


@pytest.mark.parametrize('frozen', [0, 1])
def test_complex_character_and_frozen_core_oracle(monkeypatch, frozen):
    from tests.test_chi_canonical_mp2 import spin_orbital_oracle
    ref, _ = fixture(monkeypatch, nk=3, nc=2, no=1, nv=1)
    hf = ref.hf_result
    channels = _independent_frame(hf, frozen)
    doubles = spin_orbital_oracle(hf.fitted_operator, channels)
    singles = 0.
    for spin, (c, e, n) in zip(('alpha', 'beta'), channels):
        for k in range(len(c)):
            f = getattr(hf, 'fock_'+spin)[k]
            for i in range(n):
                for a in range(n, c.shape[-1]):
                    singles += abs(c[k, :, a].conj() @ f @ c[k, :, i])**2/(e[k, i]-e[k, a])/len(c)
    result = rmp.evaluate_same_source_rohf_mp2(ref, frozen_core=frozen, auxiliary_batch_size=1)
    assert result.e_singles_per_cell == pytest.approx(singles, abs=2e-13)
    assert result.e_doubles_per_cell == pytest.approx(doubles, abs=2e-13)
    assert result.active_spaces == ((frozen, 3-frozen, 1), (frozen, 2-frozen, 2))
    assert result.source_identity == hf.fitted_operator.identity
    assert result.canonicalization_identity
    assert 'projector' in result.frozen_core_convention


@pytest.mark.parametrize('frozen', [0, 1])
def test_closed_open_virtual_rotations_and_phases(monkeypatch, frozen):
    ref, _ = fixture(monkeypatch, nk=3, nc=2, no=2, nv=2)
    expected = rmp.evaluate_same_source_rohf_mp2(ref, frozen_core=frozen)
    hf = ref.hf_result; c = hf.mo_coeffs.copy(); rng = np.random.default_rng(451)
    for k in range(len(c)):
        for lo, hi in ((0, 2), (2, 4), (4, 6)):
            u, _ = np.linalg.qr(rng.normal(size=(hi-lo, hi-lo))+1j*rng.normal(size=(hi-lo, hi-lo)))
            c[k, :, lo:hi] = c[k, :, lo:hi] @ u
    hf.mo_coeffs = hf.mo_coeffs_alpha = hf.mo_coeffs_beta = c
    result = rmp.evaluate_same_source_rohf_mp2(reseal(ref), frozen_core=frozen)
    assert result.e_corr_per_cell == pytest.approx(expected.e_corr_per_cell, abs=2e-13)
    assert result.e_singles_per_cell == pytest.approx(expected.e_singles_per_cell, abs=2e-13)


def test_closed_shell_mp2_limit(monkeypatch):
    ref, _ = fixture(monkeypatch, nk=3, nc=1, no=0, nv=2)
    hf = ref.hf_result
    rohf = rmp.evaluate_same_source_rohf_mp2(ref)
    channels = _independent_frame(hf)
    c, energies, count = channels[0]
    hf.mo_coeffs, hf.mo_energies = c, energies
    hf.fock, hf.density = hf.fock_alpha, 2*hf.density_alpha
    from tests.test_chi_canonical_mp2 import Diagnostic
    hf.aiccm2026dev_b = Diagnostic('RHF')
    rhf = mp.evaluate_same_source_mp2(reseal(replace(ref, method='RHF')))
    assert rohf.e_corr_per_cell == pytest.approx(rhf.e_corr_per_cell, abs=2e-13)
    assert abs(rohf.e_singles_per_cell) < 1e-24


@pytest.mark.parametrize('change', ['shared-orbitals', 'alpha-fock', 'subspaces', 'counts', 'canonicalization', 'factors'])
def test_mutation_rejected(monkeypatch, change):
    ref, _ = fixture(monkeypatch)
    h = ref.hf_result
    if change == 'shared-orbitals': h.mo_coeffs[0, 0, 0] += .1j
    elif change == 'alpha-fock': h.fock_alpha[0, 0, 0] += .01
    elif change == 'subspaces': h.rohf_subspaces['closed'] = (1,)
    elif change == 'counts': h.n_open = 0
    elif change == 'canonicalization': h.rohf_canonicalization = 'different'
    else:
        op = h.fitted_operator
        h.fitted_operator = ChiFittedOperator({p: a*1.01 for p, a in op.items()}, op.descriptor, op._weights)
    with pytest.raises(ValueError, match='changed'):
        rmp.evaluate_same_source_rohf_mp2(ref)


@pytest.mark.parametrize('change', ['fractional', 'stationarity', 'spin-density', 'subspaces', 'near-degenerate', 'core-degenerate', 'sr'])
def test_resealed_incompatible_state_refused(monkeypatch, change):
    ref, _ = fixture(monkeypatch, nc=2, no=1, nv=1)
    h = ref.hf_result; kwargs = {}
    if change == 'fractional': h.occupations[0, 0] = 1.5
    elif change == 'stationarity':
        x = np.outer(h.mo_coeffs[0, :, 0], h.mo_coeffs[0, :, -1].conj())*.1
        h.fock_alpha[0] += x+x.conj().T
    elif change == 'spin-density': h.density_alpha[0] *= .99
    elif change == 'subspaces': h.rohf_subspaces['open'] = (1,)
    elif change in ('near-degenerate', 'core-degenerate'):
        h.fock_alpha[:] = 0.; h.fock_beta[:] = 0.
        if change == 'core-degenerate': kwargs['frozen_core'] = 1
    else: h.executed_fit_source = replace(h.executed_fit_source, source='sr')
    with pytest.raises((ValueError, NotImplementedError)):
        rmp.evaluate_same_source_rohf_mp2(reseal(ref), **kwargs)


@pytest.mark.parametrize('core', [True, -1, (0, 1), 2, 0.5, None])
def test_invalid_frozen_core(monkeypatch, core):
    ref, _ = fixture(monkeypatch)
    with pytest.raises(ValueError):
        rmp.evaluate_same_source_rohf_mp2(ref, frozen_core=core)


def test_uhf_cannot_be_relabelled_rohf(monkeypatch):
    from tests.test_chi_canonical_mp2 import fixture as unrestricted_fixture
    ref, _ = unrestricted_fixture(monkeypatch, unrestricted=True, source='classic')
    with pytest.raises(ValueError, match='genuine ROHF'):
        rmp.evaluate_same_source_rohf_mp2(ref)


@pytest.mark.parametrize('key,value', [('smearing_temperature', .1), ('ecp_total_ncore', 2), ('executed_chi_xc', {'functional': 'test'})])
def test_unsupported_reference_refused(monkeypatch, key, value):
    ref, _ = fixture(monkeypatch)
    setattr(ref.hf_result, key, value)
    with pytest.raises((ValueError, NotImplementedError)):
        rmp.evaluate_same_source_rohf_mp2(reseal(ref))
