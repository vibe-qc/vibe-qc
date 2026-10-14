"""Independent determinant-space checks of complex CC equations."""
from itertools import combinations
import numpy as np
import pytest
from scipy.linalg import expm
from vibeqc.periodic.chi._cc_equations import residual, _ccsd_energy_so


def action(bits, holes, particles):
    sign = 1
    for p, create in [(p, False) for p in holes] + [(p, True) for p in particles]:
        if bool(bits & (1 << p)) == create:
            return None
        sign *= (-1) ** ((bits & ((1 << p)-1)).bit_count())
        bits ^= 1 << p
    return bits, sign


def determinant_reference(f, g, t1, t2):
    """Full fixed-N Hamiltonian and finite-matrix exp(-T) H exp(T)."""
    nv, no = t1.shape
    n = no + nv
    basis = [sum(1 << i for i in c) for c in combinations(range(n), no)]
    index = {b: i for i, b in enumerate(basis)}
    dim = len(basis)
    h = f - np.einsum('piqi->pq', g[:, :no, :, :no])
    H = np.zeros((dim, dim), complex)
    T = np.zeros_like(H)
    V = np.zeros_like(H)
    S = np.zeros_like(H)
    D = np.zeros_like(H)
    def add(matrix, coefficient, holes, particles):
        for col, b in enumerate(basis):
            result = action(b, holes, particles)
            if result:
                matrix[index[result[0]], col] += coefficient * result[1]
    for p in range(n):
        for q in range(n):
            add(H, h[p,q], [q], [p])
            for r in range(n):
                for s in range(n):
                    add(V, g[p,q,r,s]/4, [r,s], [q,p])
    H += V
    # Normal-ordered fluctuation operator for canonical HF.
    for p in range(n):
        for q in range(n):
            add(V, h[p,q]-f[p,q], [q], [p])
    for a in range(nv):
        for i in range(no):
            add(S, t1[a,i], [i], [no+a])
            for b in range(nv):
                for j in range(no):
                    add(D, t2[a,b,i,j]/4, [i,j], [no+b,no+a])
    T = S + D
    ref = index[(1 << no)-1]
    bar = expm(-T) @ H @ expm(T)
    r1 = np.zeros_like(t1, dtype=complex)
    r2 = np.zeros_like(t2, dtype=complex)
    for a in range(nv):
        for i in range(no):
            b, sign = action(basis[ref], [i], [no+a])
            r1[a,i] = sign * bar[index[b],ref]
            for b in range(nv):
                for j in range(no):
                    hit = action(basis[ref], [i,j], [no+b,no+a])
                    if hit:
                        r2[a,b,i,j] = hit[1] * bar[index[hit[0]],ref]
    return bar[ref,ref]-H[ref,ref], r1, r2, (V,S,D,basis,ref)


def model(seed=7, n=8, no=4):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(5,n,n)) * .03
    B = B + B.transpose(0,2,1)
    z = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    u = np.linalg.qr(z)[0]
    B = np.einsum('pi,Ppq,qj->Pij', u.conj(), B, u)
    g = np.einsum('Ppr,Pqs->pqrs',B,B)-np.einsum('Pps,Pqr->pqrs',B,B)
    f = np.diag(np.r_[np.linspace(-2,-1,no),np.linspace(.5,1.5,n-no)])
    t1 = (rng.normal(size=(n-no,no))+1j*rng.normal(size=(n-no,no))) * .015
    x = (rng.normal(size=(n-no,n-no,no,no))+1j*rng.normal(size=(n-no,n-no,no,no))) * .01
    t2 = x-x.swapaxes(0,1)-x.swapaxes(2,3)+x.swapaxes(0,1).swapaxes(2,3)
    return f,g,t1,t2


def test_arbitrary_complex_residuals_against_determinants():
    f,g,t1,t2 = model()
    energy,r1,r2,_ = determinant_reference(f,g,t1,t2)
    actual = residual(t1,t2,f,g,4)
    np.testing.assert_allclose(actual[0],r1,atol=2e-13)
    np.testing.assert_allclose(actual[1],r2,atol=2e-13)
    np.testing.assert_allclose(_ccsd_energy_so(t1,t2,f,g,4),energy,atol=2e-13)


def test_nonzero_complex_triples_against_determinants():
    from vibeqc.periodic.chi._cc_equations import triples
    f,g,t1,t2 = (x.real for x in model())
    _,_,_,(V,S,D,basis,ref) = determinant_reference(f,g,t1,t2)
    right, single = (V @ D)[:,ref], (V @ S)[:,ref]
    expected = 0j
    for idx,bits in enumerate(basis):
        if (bits & 15).bit_count() == 1:
            den = sum(f[i,i] for i in range(4) if not bits&(1<<i))-sum(f[a,a] for a in range(4,8) if bits&(1<<a))
            expected += right[idx].conjugate()*(right[idx]+single[idx])/den
    assert abs(expected)>1e-8
    np.testing.assert_allclose(triples(t1,t2,np.diag(f),g),expected,atol=2e-13)
    phase = np.exp(1j*np.arange(8)*.71)
    gg = np.einsum('p,q,r,s,pqrs->pqrs',phase.conj(),phase.conj(),phase,phase,g)
    tt1 = np.einsum('a,i,ai->ai',phase[4:].conj(),phase[:4],t1)
    tt2 = np.einsum('a,b,i,j,abij->abij',phase[4:].conj(),phase[4:].conj(),phase[:4],phase[:4],t2)
    np.testing.assert_allclose(triples(tt1,tt2,np.diag(f),gg),expected,atol=2e-13)


def test_factor_blocks_complex_momentum_and_mp2_limit(monkeypatch):
    from tests.test_chi_canonical_mp2 import fixture
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter, evaluate_same_source_mp2
    from vibeqc.periodic.chi.canonical_cc import FactorIntegrals
    ref,channels = fixture(monkeypatch,nk=3,source='classic',nao=2)
    c,e,no = channels[0]
    blocks = FactorIntegrals(CanonicalFactorAdapter(ref.hf_result.fitted_operator),c,no,0,1)
    labels = blocks.labels
    n = len(labels)
    actual = blocks[(slice(None),)*4]
    expected = np.zeros_like(actual)
    op = ref.hf_result.fitted_operator
    for p,(kp,mp,sp) in enumerate(labels):
        for q,(kq,mq,sq) in enumerate(labels):
            for r,(kr,mr,sr) in enumerate(labels):
                for s,(ks,ms,ss) in enumerate(labels):
                    if (kp+kq-kr-ks)%3:
                        continue
                    def chem(ka,ma,kb,mb,kc,mc,kd,md):
                        ao = np.einsum('Puv,Pyx->uvxy',op[ka,kb],op[kd,kc].conj())
                        return np.einsum('u,v,x,y,uvxy->',c[ka,:,ma].conj(),c[kb,:,mb],c[kc,:,mc].conj(),c[kd,:,md],ao)/3
                    if sp==sr and sq==ss:
                        expected[p,q,r,s]+=chem(kp,mp,kr,mr,kq,mq,ks,ms)
                    if sp==ss and sq==sr:
                        expected[p,q,r,s]-=chem(kp,mp,ks,ms,kq,mq,kr,mr)
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    eps = np.array([e[k,p] for k,p,s in labels])
    no = blocks.no
    d = eps[:no][None,None,:,None]+eps[:no][None,None,None,:]-eps[no:][:,None,None,None]-eps[no:][None,:,None,None]
    energy = .25*np.einsum('ijab,abij->',actual[:no,:no,no:,no:],actual[no:,no:,:no,:no]/d)/3
    np.testing.assert_allclose(energy,evaluate_same_source_mp2(ref).e_corr_per_cell,atol=1e-14)


def test_solver_two_electron_exact_and_refusals():
    import pytest
    from vibeqc.periodic.chi.canonical_cc import solve,ChiCCOptions
    class Dense:
        def __init__(self,g,no): self.g,self.no,self.shape=g,no,g.shape
        def __getitem__(self,key): return self.g[key]
    f,g,t1,t2 = model(n=6,no=2)
    options=ChiCCOptions(residual_tolerance=1e-10,energy_tolerance=1e-12)
    energy,t1,t2,trace,_ = solve(Dense(g,2),np.diag(f),options)
    ec,r1,r2,_=determinant_reference(f,g,t1,t2)
    np.testing.assert_allclose(energy,ec,atol=1e-12)
    assert max(abs(r1).max(),abs(r2).max())<1e-10
    # CCSD spans the exact two-electron wavefunction; check its CI eigenvalue.
    _,_,_,(V,S,D,basis,ref)=determinant_reference(f,g,t1,t2)
    H=V.copy()
    for col,bits in enumerate(basis):
        H[col,col]+=sum(f[i,i] for i in range(6) if bits&(1<<i))
    np.testing.assert_allclose(energy,np.linalg.eigvalsh(H)[0]-H[ref,ref],atol=1e-11)
    with pytest.raises(RuntimeError,match='did not converge'):
        solve(Dense(g,2),np.diag(f),ChiCCOptions(max_iter=1))
    with pytest.raises(MemoryError):
        solve(Dense(g,2),np.diag(f),ChiCCOptions(memory_byte_cap=1))
    with pytest.raises(ValueError,match='denominator'):
        solve(Dense(g,2),np.zeros(6),options)


def test_degenerate_rotations_and_fragment_additivity():
    from vibeqc.periodic.chi.canonical_cc import solve,ChiCCOptions
    from vibeqc.periodic.chi._cc_equations import triples
    class Dense:
        def __init__(self,g,no): self.g,self.no,self.shape=g,no,g.shape
        def __getitem__(self,key): return self.g[key]
    f,g,_,_ = model(n=4,no=2)
    eps=np.array([-1.,-1.,1.,1.])
    opts=ChiCCOptions(energy_tolerance=1e-12,residual_tolerance=1e-10)
    energy,t1,t2,_,_=solve(Dense(g,2),eps,opts)
    assert triples(t1,t2,eps,g)==0.
    rng=np.random.default_rng(44)
    u=np.zeros((4,4),complex)
    for s in (slice(0,2),slice(2,4)):
        u[s,s]=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
    gg=np.einsum('pa,qb,rc,sd,pqrs->abcd',u.conj(),u.conj(),u,u,g)
    np.testing.assert_allclose(solve(Dense(gg,2),eps,opts)[0],energy,atol=1e-11)
    pair=np.zeros((8,)*4,complex)
    for labels in ([0,1,4,5],[2,3,6,7]): pair[np.ix_(labels,labels,labels,labels)]=g
    doubled=solve(Dense(pair,4),np.r_[-np.ones(4),np.ones(4)],opts)
    np.testing.assert_allclose(doubled[0],2*energy,atol=1e-11)
    np.testing.assert_allclose(triples(doubled[1],doubled[2],np.r_[-np.ones(4),np.ones(4)],pair),0.,atol=1e-15)


def test_public_cc_capabilities_and_output():
    import pytest
    from vibeqc.periodic.chi.correlation import require_public_cc_admission
    assert 'experimental' in require_public_cc_admission('RHF','classic')
    assert 'experimental' in require_public_cc_admission('UHF','classic')
    for method,source in [('ROHF','classic'),('RKS','classic'),('UKS','classic'),('RHF','sr'),('UHF','sr'),('RHF','auto')]:
        with pytest.raises(NotImplementedError): require_public_cc_admission(method,source)


def test_public_cc_controls_pass_full_preflight(tmp_path):
    import tomllib
    import vibeqc as vq
    from vibeqc.periodic.chi.fitted import ChiCoulombControls
    from vibeqc.periodic.chi.canonical_cc import ChiCCOptions
    cell=vq.PeriodicSystem(3,np.eye(3)*8,
        [vq.Atom(1,[1.,1.,1.]),vq.Atom(1,[1.,1.,2.4])],0,1)
    basis=vq.BasisSet(cell.unit_cell_molecule(),'sto-3g')
    stem=tmp_path/'cc-preflight'
    vq.run_periodic_job(cell,basis,method='aiccm',variant='chi',correlation='ccsd',
        scf_reference='rhf',fit_source='classic',aiccm_backend='ri',
        chi_coulomb_controls=ChiCoulombControls(6.2,6.2,18.,8.,5.,.6),
        chi_cc_options=ChiCCOptions(),conv_tol_energy=1e-11,conv_tol_grad=1e-9,
        kpoints=(1,1,1),dry_run=True,output=stem,output_qvf=False,record_hostname=False)
    record=tomllib.loads(stem.with_suffix('.system').read_text())
    assert record['outputs']['status']=='dry_run'


def test_weak_coupling_limit():
    from vibeqc.periodic.chi.canonical_cc import solve, ChiCCOptions
    class Dense:
        def __init__(self,g): self.g,self.no,self.shape=g,2,g.shape
        def __getitem__(self,key): return self.g[key]
    f,g,_,_=model(n=4,no=2)
    f=np.diag([-1.,-1.,1.,1.]);g=g.real
    errors=[]
    for coupling in (1e-2,1e-3):
        scaled=g*coupling
        energy=solve(Dense(scaled),np.diag(f),ChiCCOptions(
            energy_tolerance=1e-16,residual_tolerance=1e-14))[0]
        mp2=.25*np.einsum('ijab,abij->',scaled[:2,:2,2:,2:],scaled[2:,2:,:2,:2]/-4).real
        errors.append(abs(energy/mp2-1))
    assert errors[1] < errors[0]/5


def test_frozen_labels_and_panel_batches(monkeypatch):
    from tests.test_chi_canonical_mp2 import fixture
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.periodic.chi.canonical_cc import FactorIntegrals
    ref,channels=fixture(monkeypatch,nk=3,source='classic',nao=4)
    c,_,_=channels[0]
    adapter=CanonicalFactorAdapter(ref.hf_result.fitted_operator)
    all_occ=FactorIntegrals(adapter,c,2,0,1)
    frozen=FactorIntegrals(adapter,c,2,1,7)
    assert frozen.no==6
    assert all(p==1 for k,p,spin in frozen.labels[:6])
    selected=[all_occ.labels.index(label) for label in frozen.labels]
    # Restrict the independently specified full active-space tensor.
    dense=all_occ[(slice(None),)*4]
    np.testing.assert_allclose(frozen[(slice(None),)*4],
        dense[np.ix_(selected,selected,selected,selected)],atol=2e-14)


# These finite Hamiltonians check equations and source transport only. They do
# not constitute native UHF scientific acceptance or a comparison with ORCA.
def unrestricted_model(occupied=(2, 1), virtual=(2, 3), seed=819):
    """Real collinear UHF Hamiltonian with independently chosen spin spaces."""
    rng = np.random.default_rng(seed)
    spins = np.array([s for counts in (occupied, virtual)
                      for s, count in enumerate(counts) for _ in range(count)])
    no, nv = sum(occupied), sum(virtual)
    n = no + nv
    factors = rng.normal(size=(5, n, n)) * .045
    factors += factors.transpose(0, 2, 1)
    factors *= spins[:, None] == spins[None, :]
    g = (np.einsum('Ppr,Pqs->pqrs', factors, factors)
         - np.einsum('Pps,Pqr->pqrs', factors, factors))
    eps = np.r_[np.linspace(-2., -1., no), np.linspace(.5, 1.5, nv)]
    t1 = rng.normal(size=(nv, no)) * .02
    t1 *= spins[no:, None] == spins[None, :no]
    x = rng.normal(size=(nv, nv, no, no)) * .015
    t2 = x-x.swapaxes(0, 1)-x.swapaxes(2, 3)+x.swapaxes(0, 1).swapaxes(2, 3)
    t2 *= (spins[no:, None, None, None]+spins[None, no:, None, None]
           == spins[None, None, :no, None]+spins[None, None, None, :no])
    return np.diag(eps), g, t1, t2, spins


def determinant_triples(f, g, t1, t2):
    """Unique determinant sum of <triple|V T2> and <triple|V T1>.

    Builds fermion operators, with no production permutation or contraction
    helper. Three occupied electrons make every triple moment connected.
    """
    _, _, _, (V, S, D, basis, ref) = determinant_reference(f, g, t1, t2)
    no = t1.shape[1]
    occupied_mask = (1 << no)-1
    right, single = (V @ D)[:, ref], (V @ S)[:, ref]
    energy = 0j
    for idx, bits in enumerate(basis):
        if (bits & occupied_mask).bit_count() != no-3:
            continue
        den = (sum(f[i, i] for i in range(no) if not bits & (1 << i))
               - sum(f[a, a] for a in range(no, len(f)) if bits & (1 << a)))
        energy += right[idx].conjugate() * (right[idx]+single[idx]) / den
    return energy


@pytest.mark.parametrize('occupied,virtual', [
    ((3, 0), (3, 1)), ((2, 1), (2, 3)),
    ((1, 2), (3, 2)), ((0, 3), (1, 3)),
], ids=['aaa', 'aab', 'abb', 'bbb'])
def test_unrestricted_nonzero_residuals_and_each_triples_sector(occupied, virtual):
    from vibeqc.periodic.chi._cc_equations import triples
    f, g, t1, t2, spins = unrestricted_model(occupied, virtual)
    no = sum(occupied)
    expected = determinant_triples(f, g, t1, t2)
    assert abs(expected) > 1e-9
    np.testing.assert_allclose(triples(t1, t2, np.diag(f), g), expected, atol=3e-13)
    # Complex phases exercise conjugation without changing the real physical
    # Hamiltonian or making an arbitrary trial triples functional complex.
    phases = np.exp(1j*np.arange(len(f))*.63)
    g = np.einsum('p,q,r,s,pqrs->pqrs', phases.conj(), phases.conj(), phases, phases, g)
    t1 = np.einsum('a,i,ai->ai', phases[no:].conj(), phases[:no], t1)
    t2 = np.einsum('a,b,i,j,abij->abij', phases[no:].conj(), phases[no:].conj(), phases[:no], phases[:no], t2)
    energy, r1, r2, _ = determinant_reference(f, g, t1, t2)
    actual = residual(t1, t2, f, g, no)
    np.testing.assert_allclose(actual[0], r1, atol=3e-13)
    np.testing.assert_allclose(actual[1], r2, atol=3e-13)
    np.testing.assert_allclose(_ccsd_energy_so(t1, t2, f, g, no), energy, atol=3e-13)
    np.testing.assert_allclose(triples(t1, t2, np.diag(f), g), expected, atol=3e-13)
    np.testing.assert_allclose(determinant_triples(f, g, t1, t2), expected, atol=3e-13)
    # Reordering within the occupied/virtual sets can interleave spin labels.
    permutation = np.r_[np.arange(no)[::-1], no+np.arange(len(f)-no)[::-1]]
    oi, vi = permutation[:no], permutation[no:]-no
    permuted_g = g[np.ix_(permutation, permutation, permutation, permutation)]
    permuted_t1, permuted_t2 = t1[np.ix_(vi, oi)], t2[np.ix_(vi, vi, oi, oi)]
    np.testing.assert_allclose(triples(permuted_t1, permuted_t2, np.diag(f)[permutation], permuted_g), expected, atol=3e-13)
    forbidden = spins[no:, None] != spins[None, :no]
    np.testing.assert_allclose(r1[forbidden], 0., atol=3e-13)


def dense_uhf_factors(operator, channels, labels):
    """Independent four-AO expansion, then four-coefficient transformation.

    Spin and character selection are explicit; this does not use production
    pair-factor transforms, momentum maps, panels, or residual equations.
    """
    nk = len(channels[0][0])
    chem = {}
    for sp in (0, 1):
        cp = channels[sp][0]
        for sq in (0, 1):
            cq = channels[sq][0]
            for kp in range(nk):
                for kr in range(nk):
                    for kq in range(nk):
                        for ks in range(nk):
                            if (kp+kq-kr-ks) % nk:
                                continue
                            ao = np.einsum('Puv,Pyx->uvxy', operator[kp, kr], operator[ks, kq].conj())
                            chem[sp, sq, kp, kr, kq, ks] = np.einsum(
                                'up,vr,xq,ys,uvxy->prqs', cp[kp].conj(), cp[kr],
                                cq[kq].conj(), cq[ks], ao, optimize=True)/nk
    g = np.zeros((len(labels),)*4, complex)
    for p, (kp, mp, sp) in enumerate(labels):
        for q, (kq, mq, sq) in enumerate(labels):
            for r, (kr, mr, sr) in enumerate(labels):
                for s, (ks, ms, ss) in enumerate(labels):
                    if (kp+kq-kr-ks) % nk:
                        continue
                    if sp == sr and sq == ss:
                        g[p, q, r, s] += chem[sp, sq, kp, kr, kq, ks][mp, mr, mq, ms]
                    if sp == ss and sq == sr:
                        g[p, q, r, s] -= chem[sp, sq, kp, ks, kq, kr][mp, ms, mq, mr]
    return g


def spin_momentum_amplitudes(labels, no, nk, seed=918):
    rng = np.random.default_rng(seed)
    k, _, spin = np.array(labels).T
    nv = len(labels)-no
    allowed1 = ((k[no:, None]-k[None, :no]) % nk == 0)
    allowed1 &= spin[no:, None] == spin[None, :no]
    allowed2 = ((k[no:, None, None, None]+k[None, no:, None, None]
                 - k[None, None, :no, None]-k[None, None, None, :no]) % nk == 0)
    allowed2 &= (spin[no:, None, None, None]+spin[None, no:, None, None]
                 == spin[None, None, :no, None]+spin[None, None, None, :no])
    t1 = (rng.normal(size=(nv, no))+1j*rng.normal(size=(nv, no)))*.01*allowed1
    x = (rng.normal(size=(nv, nv, no, no))+1j*rng.normal(size=(nv, nv, no, no)))*.01
    t2 = (x-x.swapaxes(0, 1)-x.swapaxes(2, 3)+x.swapaxes(0, 1).swapaxes(2, 3))*allowed2
    return t1, t2, allowed1, allowed2


@pytest.mark.parametrize('nk,nao', [(1, 4), (3, 3)])
def test_uhf_factor_transport_and_blocked_residuals(monkeypatch, nk, nao):
    from tests.test_chi_canonical_mp2 import fixture
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter, evaluate_same_source_mp2
    from vibeqc.periodic.chi.canonical_cc import FactorIntegrals
    reference, channels = fixture(monkeypatch, nk=nk, nao=nao, unrestricted=True, source='classic')
    adapter = CanonicalFactorAdapter(reference.hf_result.fitted_operator)
    blocks = FactorIntegrals(adapter, tuple(c for c, e, n in channels),
                            tuple(n for c, e, n in channels), (0, 0), 1)
    labels, no = blocks.labels, blocks.no
    assert no == 3*nk
    assert [sum(s == spin for k, p, s in labels[:no]) for spin in (0, 1)] == [2*nk, nk]
    assert [sum(s == spin for k, p, s in labels[no:]) for spin in (0, 1)] == [(nao-2)*nk, (nao-1)*nk]
    expected = dense_uhf_factors(adapter.operator, channels, labels)
    np.testing.assert_allclose(blocks[(slice(None),)*4], expected, atol=3e-14)
    eps = np.array([channels[s][1][k, p] for k, p, s in labels])
    t1, t2, allowed1, allowed2 = spin_momentum_amplitudes(labels, no, nk)
    actual = residual(t1, t2, np.diag(eps), blocks, no)
    dense = residual(t1, t2, np.diag(eps), expected, no)
    np.testing.assert_allclose(actual[0], dense[0], atol=3e-13)
    np.testing.assert_allclose(actual[1], dense[1], atol=3e-13)
    np.testing.assert_allclose(actual[0][~allowed1], 0., atol=3e-13)
    np.testing.assert_allclose(actual[1][~allowed2], 0., atol=3e-13)
    if nk == 1:
        energy, r1, r2, _ = determinant_reference(np.diag(eps), expected, t1, t2)
        np.testing.assert_allclose(actual[0], r1, atol=3e-13)
        np.testing.assert_allclose(actual[1], r2, atol=3e-13)
        np.testing.assert_allclose(_ccsd_energy_so(t1, t2, np.diag(eps), blocks, no), energy, atol=3e-13)
    else:
        k = np.array(labels)[:, 0]
        forbidden = ((k[:, None, None, None]+k[None, :, None, None]
                      - k[None, None, :, None]-k[None, None, None, :]) % nk != 0)
        assert np.count_nonzero(expected[~forbidden].imag) > 0
        assert np.count_nonzero(blocks[(slice(None),)*4][forbidden]) == 0
    d1 = eps[:no][None, :]-eps[no:][:, None]
    d2 = d1[:, None, :, None]+d1[None, :, None, :]
    mp2 = .25*np.einsum('ijab,abij->', expected[:no, :no, no:, no:], expected[no:, no:, :no, :no]/d2)/nk
    np.testing.assert_allclose(mp2, evaluate_same_source_mp2(reference).e_corr_per_cell, atol=3e-14)


def test_uhf_frozen_spin_indexing_and_auxiliary_batches(monkeypatch):
    from tests.test_chi_canonical_mp2 import fixture
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.periodic.chi.canonical_cc import FactorIntegrals
    reference, channels = fixture(monkeypatch, nk=3, nao=3, unrestricted=True, source='classic')
    adapter = CanonicalFactorAdapter(reference.hf_result.fitted_operator)
    coefficients, occupied = tuple(c for c, e, n in channels), (2, 1)
    full = FactorIntegrals(adapter, coefficients, occupied, (0, 0), 1)
    frozen = FactorIntegrals(adapter, coefficients, occupied, (1, 0), 7)
    assert frozen.no == 6
    assert set(frozen.labels[:6]) == {(k, p, s) for k in range(3) for p, s in ((1, 0), (0, 1))}
    keep = [full.labels.index(label) for label in frozen.labels]
    oracle = dense_uhf_factors(adapter.operator, channels, full.labels)
    np.testing.assert_allclose(frozen[(slice(None),)*4], oracle[np.ix_(keep, keep, keep, keep)], atol=3e-14)


class DenseIntegrals:
    def __init__(self, g, no):
        self.g, self.no, self.shape = g, no, g.shape

    def __getitem__(self, key):
        return self.g[key]


def test_uhf_weak_coupling_and_two_electron_limits():
    from vibeqc.periodic.chi.canonical_cc import solve, ChiCCOptions
    from vibeqc.periodic.chi._cc_equations import triples
    opts = ChiCCOptions(energy_tolerance=1e-14, residual_tolerance=1e-12)
    f, g, _, _, _ = unrestricted_model(occupied=(1, 1), virtual=(1, 2))
    energy, t1, t2, _, _ = solve(DenseIntegrals(g, 2), np.diag(f), opts)
    ec, r1, r2, (V, _, _, basis, ref) = determinant_reference(f, g, t1, t2)
    H = V.copy()
    for col, bits in enumerate(basis):
        H[col, col] += sum(f[i, i] for i in range(len(f)) if bits & (1 << i))
    np.testing.assert_allclose(energy, np.linalg.eigvalsh(H)[0]-H[ref, ref], atol=3e-12)
    np.testing.assert_allclose(energy, ec, atol=3e-12)
    assert max(abs(r1).max(), abs(r2).max()) < 1e-12
    assert triples(t1, t2, np.diag(f), g) == 0.
    f, g, _, _, _ = unrestricted_model()
    eps, no = np.diag(f), 3
    d1 = eps[:no][None, :]-eps[no:][:, None]
    d2 = d1[:, None, :, None]+d1[None, :, None, :]
    errors = []
    for coupling in (1e-2, 1e-3):
        scaled = coupling*g
        ecc = solve(DenseIntegrals(scaled, no), eps, opts)[0]
        mp2 = .25*np.einsum('ijab,abij->', scaled[:no, :no, no:, no:], scaled[no:, no:, :no, :no]/d2)
        errors.append(abs(ecc/mp2-1))
    assert errors[1] < errors[0]/5


def test_uhf_nonzero_triples_fragment_additivity():
    from vibeqc.periodic.chi.canonical_cc import solve, ChiCCOptions
    from vibeqc.periodic.chi._cc_equations import triples
    f, g, _, _, _ = unrestricted_model(virtual=(2, 2))
    g *= 2.
    eps, no, nv = np.diag(f), 3, 4
    opts = ChiCCOptions(energy_tolerance=1e-13, residual_tolerance=1e-11)
    energy, t1, t2, _, _ = solve(DenseIntegrals(g, no), eps, opts)
    et = triples(t1, t2, eps, g)
    assert abs(et) > 1e-10
    np.testing.assert_allclose(et, determinant_triples(f, g, t1, t2), atol=3e-13)
    pair = np.zeros((2*len(f),)*4)
    for occupied, virtual in ((range(no), range(2*no, 2*no+nv)),
                              (range(no, 2*no), range(2*no+nv, 2*(no+nv)))):
        ix = list(occupied)+list(virtual)
        pair[np.ix_(ix, ix, ix, ix)] = g
    both_eps = np.r_[eps[:no], eps[:no], eps[no:], eps[no:]]
    doubled = solve(DenseIntegrals(pair, 2*no), both_eps, opts)
    np.testing.assert_allclose(doubled[0], 2*energy, atol=3e-12)
    np.testing.assert_allclose(triples(doubled[1], doubled[2], both_eps, pair), 2*et, atol=3e-12)


def replace_reference_channels(reference, channels):
    """Specify a consistent synthetic HF state before resealing it."""
    from tests.test_chi_canonical_mp2 import reseal
    hf = reference.hf_result
    unrestricted = reference.method == 'UHF'
    for spin, (c, e, n) in enumerate(channels if unrestricted else channels[:1]):
        suffix = ('_alpha', '_beta')[spin] if unrestricted else ''
        occupancy = 1 if unrestricted else 2
        occ = np.zeros_like(e)
        occ[:, :n] = occupancy
        for field, value in (
            ('mo_coeffs', c), ('mo_energies', e), ('occupations', occ),
            ('fock', (c*e[:, None, :])@c.conj().transpose(0, 2, 1)),
            ('density', occupancy*c[:, :, :n]@c[:, :, :n].conj().transpose(0, 2, 1)),
        ):
            setattr(hf, field+suffix, value)
    return reseal(reference)


def test_same_root_uhf_closed_shell_ccsd_and_nonzero_triples(monkeypatch):
    from tests.test_chi_canonical_mp2 import fixture
    from vibeqc.periodic.chi.canonical_cc import evaluate_same_source_cc, ChiCCOptions
    rng = np.random.default_rng(754)
    c = np.linalg.qr(rng.normal(size=(4, 4)))[0][None, :, :]
    e = np.array([[-2., -1., .5, 1.5]])
    channels = [(c, e, 2), (c.copy(), e.copy(), 2)]
    results = []
    for unrestricted in (False, True):
        ref, _ = fixture(monkeypatch, nk=1, nao=4, unrestricted=unrestricted, source='classic')
        ref = replace_reference_channels(ref, channels)
        results.append(evaluate_same_source_cc(ref, options=ChiCCOptions(
            energy_tolerance=1e-13, residual_tolerance=1e-11)))
    restricted, unrestricted = results
    assert abs(restricted.e_triples_per_cell) > 1e-10
    np.testing.assert_allclose(unrestricted.e_ccsd_per_cell, restricted.e_ccsd_per_cell, atol=3e-12)
    np.testing.assert_allclose(unrestricted.e_triples_per_cell, restricted.e_triples_per_cell, atol=3e-12)
    np.testing.assert_allclose(unrestricted.t1, restricted.t1, atol=3e-12)
    np.testing.assert_allclose(unrestricted.t2, restricted.t2, atol=3e-12)
    assert unrestricted.active_spaces == ((0, 2, 2), (0, 2, 2))
    assert unrestricted.source_identity == restricted.source_identity
    assert unrestricted.quantitative_status == 'not-qualified'


def test_uhf_unequal_total_orbital_dimensions(monkeypatch):
    from tests.test_chi_canonical_mp2 import fixture
    from vibeqc.periodic.chi.canonical_cc import FactorIntegrals, evaluate_same_source_cc
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    ref, channels = fixture(monkeypatch, nk=1, nao=4, unrestricted=True, source='classic')
    rng = np.random.default_rng(399)
    channels = [(np.linalg.qr(rng.normal(size=(4, 4)))[0][None]
                 * np.exp(1j*np.arange(4)*(.31+.17*s))[None, None, :], e, n)
                for s, (c, e, n) in enumerate(channels)]
    cb, eb, nb = channels[1]
    channels[1] = (cb[:, :, :3], eb[:, :3], nb)
    ref = replace_reference_channels(ref, channels)
    integrals = FactorIntegrals(CanonicalFactorAdapter(ref.hf_result.fitted_operator),
        tuple(c for c, e, n in channels), (2, 1), (0, 0), 1)
    assert integrals.shape == (7,)*4
    dense = dense_uhf_factors(ref.hf_result.fitted_operator, channels, integrals.labels)
    np.testing.assert_allclose(integrals[(slice(None),)*4], dense, atol=3e-14)
    result = evaluate_same_source_cc(ref, compute_triples=False)
    assert result.active_spaces == ((0, 2, 2), (0, 1, 2))
    eps = np.array([channels[s][1][k, p] for k, p, s in integrals.labels])
    energy, r1, r2, _ = determinant_reference(np.diag(eps), dense, result.t1, result.t2)
    np.testing.assert_allclose(result.e_ccsd_per_cell, energy, atol=3e-12)
    assert max(abs(r1).max(), abs(r2).max()) < 1e-8


def test_uhf_forbidden_denominators_and_allowed_triples_refusal(monkeypatch):
    from tests.test_chi_canonical_mp2 import fixture
    from vibeqc.periodic.chi.canonical_cc import FactorIntegrals, solve, ChiCCOptions
    from vibeqc.periodic.chi.canonical_mp2 import CanonicalFactorAdapter
    from vibeqc.periodic.chi._cc_equations import triples
    ref, channels = fixture(monkeypatch, nk=1, nao=4, unrestricted=True, source='classic')
    # Real frames keep the named triples expression real at a physical root.
    rng = np.random.default_rng(257)
    coefficients = tuple(np.linalg.qr(rng.normal(size=(4, 4)))[0][None] for _ in (0, 1))
    blocks = FactorIntegrals(CanonicalFactorAdapter(ref.hf_result.fitted_operator),
                            coefficients, (2, 1), (0, 0), 2)
    eps = np.array([-2.+1.2*p for k, p, s in blocks.labels])
    spins = np.array([s for k, p, s in blocks.labels])
    opts = ChiCCOptions(energy_tolerance=1e-13, residual_tolerance=1e-11)
    original = solve(blocks, eps, opts)
    et = triples(original[1], original[2], eps, blocks)
    assert abs(et) > 1e-12
    for shift in (2.4, 12.):
        moved_eps = eps+shift*spins
        # Cross-spin excitations include zero or positive gaps, but every
        # spin-conserving excitation denominator is unchanged.
        assert np.max(moved_eps[:blocks.no][None, :]-moved_eps[blocks.no:][:, None]) >= -1e-14
        moved = solve(blocks, moved_eps, opts)
        np.testing.assert_allclose(moved[0], original[0], atol=3e-12)
        np.testing.assert_allclose(moved[1], original[1], atol=3e-12)
        np.testing.assert_allclose(moved[2], original[2], atol=3e-12)
        np.testing.assert_allclose(triples(moved[1], moved[2], moved_eps, blocks), et, atol=3e-12)
    with pytest.raises(ValueError, match='denominator'):
        triples(original[1], original[2], np.zeros_like(eps), blocks)


@pytest.mark.parametrize('failure', ['swapped', 'resealed-swapped', 'fractional', 'incomplete', 'sr'])
def test_uhf_cc_reference_and_convergence_failures(monkeypatch, failure):
    from tests.test_chi_canonical_mp2 import fixture, reseal
    from vibeqc.periodic.chi.canonical_cc import evaluate_same_source_cc, ChiCCOptions
    ref, _ = fixture(monkeypatch, nk=1, nao=4, unrestricted=True,
                     source='sr' if failure == 'sr' else 'classic')
    hf = ref.hf_result
    if failure in ('swapped', 'resealed-swapped'):
        hf.mo_coeffs_alpha, hf.mo_coeffs_beta = hf.mo_coeffs_beta, hf.mo_coeffs_alpha
        if failure == 'resealed-swapped':
            ref = reseal(ref)
        with pytest.raises(ValueError):
            evaluate_same_source_cc(ref, compute_triples=False)
    elif failure == 'fractional':
        hf.occupations_beta[0, 0] = .5
        with pytest.raises(ValueError, match='occupation'):
            evaluate_same_source_cc(reseal(ref), compute_triples=False)
    elif failure == 'incomplete':
        with pytest.raises(RuntimeError, match='did not converge'):
            evaluate_same_source_cc(ref, compute_triples=False, options=ChiCCOptions(max_iter=1))
    else:
        with pytest.raises(NotImplementedError, match='SR'):
            evaluate_same_source_cc(ref, compute_triples=False)


def test_non_gamma_complex_uhf_triples_momentum_selection():
    from vibeqc.periodic.chi._cc_equations import triples
    f, g, _, _, spins = unrestricted_model()
    no, nk = 3, 3
    k = np.array([0, 1, 2, 0, 2, 0, 1, 2])
    labels = [(int(ki), p, int(s)) for p, (ki, s) in enumerate(zip(k, spins))]
    conserved = ((k[:, None, None, None]+k[None, :, None, None]
                  - k[None, None, :, None]-k[None, None, None, :]) % nk == 0)
    g *= conserved
    # A phase rotation preserves Hermiticity and all transfer characters.
    phase = np.exp(1j*np.arange(len(f))*.37)
    g = np.einsum('p,q,r,s,pqrs->pqrs', phase.conj(), phase.conj(), phase, phase, g)
    t1, t2, _, _ = spin_momentum_amplitudes(labels, no, nk)
    t1[:] = 0.
    expected = determinant_triples(f, g, t1, t2)
    assert abs(expected) > 1e-10
    np.testing.assert_allclose(triples(t1, t2, np.diag(f), g), expected, atol=3e-13)

    class MomentumIntegrals(DenseIntegrals):
        def allowed_excitation(self, occupied, virtual):
            return (sum(spins[i] for i in occupied) == sum(spins[a] for a in virtual)
                    and (sum(k[i] for i in occupied)-sum(k[a] for a in virtual)) % nk == 0)

    blocked = MomentumIntegrals(g, no)
    np.testing.assert_allclose(triples(t1, t2, np.diag(f), blocked), expected, atol=3e-13)
    # Only aab is possible; the beta k=1 virtual is the sole allowed triple.
    # Make the forbidden beta k=0 triple exactly singular without changing it.
    shifted = np.diag(f).copy()
    shifted[5] = sum(shifted[:3])-shifted[3]-shifted[4]
    np.testing.assert_allclose(triples(t1, t2, shifted, blocked), expected, atol=3e-13)
    with pytest.raises(ValueError, match='denominator'):
        triples(t1, t2, shifted, g)
