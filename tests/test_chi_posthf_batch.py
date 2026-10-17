"""Bounded classic chi post-HF batching, without a construction oracle."""
from __future__ import annotations

from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from vibeqc.periodic.chi import posthf as p

pytestmark = pytest.mark.experimental


@pytest.fixture
def fixture(monkeypatch):
    # Includes a non-self-conjugate character and reciprocal relabels.
    kcart = np.array([[0., 0., 0.], [1/3, 0., 0.], [2/3, 0., 0.]])
    system = SimpleNamespace(lattice=2*np.pi*np.eye(3), unit_cell_molecule=lambda: None)
    basis = SimpleNamespace(nbasis=3)
    aux = SimpleNamespace(nbasis=2)
    monkeypatch.setattr(p, "make_aux_basis_set", lambda *a, **kw: aux)
    monkeypatch.setattr(p, "make_modrho_aux_basis", lambda *a: aux)
    calls, refs = [], []

    def shared(system, basis, modrho, bras, q, **kw):
        # The streamed consumer must release the entire previous AO batch.
        assert all(r() is None for r in refs)
        assert kw["canonical_auxiliary_basis"] is True
        assert kw["lat_opts"].cutoff_bohr == 23.
        assert kw["ke_cutoff"] == 321.
        assert kw["linear_dep_thr"] == 1e-8
        assert modrho is aux
        calls.append((bras.copy(), q.copy(), kw["_q_metric_cache"]))
        out = []
        for bra in bras:
            i = int(round(bra[0]*3))
            j = int(round((bra[0]+q[0])*3)) % 3
            factor = np.full((2, 3, 3), 10*i+j+1j, dtype=complex)
            out.append(factor)
            refs.append(weakref.ref(factor))
        return out

    monkeypatch.setattr(p, "build_lpq_bloch_native_fft_shared_q", shared)
    controls = dict(lattice_cutoff_bohr=23., rsgdf_ke_cutoff=321., gdf_linear_dep_threshold=1e-8)
    return system, basis, aux, kcart, controls, calls, refs


@pytest.mark.parametrize("width", [1, 2, 3])
def test_streamed_batches_release_ao_and_preserve_all_pairs(fixture, width):
    system, basis, aux, kcart, controls, calls, refs = fixture
    coeffs = [np.eye(3, dtype=complex) for _ in kcart]
    lov, got_aux = p._build_canonical_lov_cache(
        system, basis, kcart, coeffs, 1, "aux", **controls,
        batch_memory_bytes=width*3*16*2*3**2,
    )
    assert got_aux is aux
    assert list(lov) == [(i, j) for i in range(3) for j in range(3)]
    assert sum(len(bras) for bras, q, metric in calls) == 9
    assert max(len(bras) for bras, q, metric in calls) <= width
    assert all(r() is None for r in refs)
    assert len({id(metric) for bras, q, metric in calls}) == 3
    for (i, j), value in lov.items():
        np.testing.assert_array_equal(value, np.full((2, 1, 2), 10*i+j+1j))


def test_ao_cache_order_and_single_pair_budget_floor(fixture, monkeypatch):
    system, basis, aux, kcart, controls, calls, refs = fixture
    original = p.build_lpq_bloch_native_fft_shared_q

    def shared(*args, **kwargs):
        refs.clear()  # AO cache deliberately retains factors.
        return original(*args, **kwargs)

    monkeypatch.setattr(p, "build_lpq_bloch_native_fft_shared_q", shared)
    cache, got_aux = p._build_canonical_lpq_cache(
        system, basis, kcart, "aux", **controls, batch_memory_bytes=1)
    assert got_aux is aux
    assert list(cache) == [(i, j) for i in range(3) for j in range(3)]
    assert len(calls) == 9
    assert all(len(bras) == 1 for bras, q, metric in calls)
    for (i, j), factor in cache.items():
        np.testing.assert_array_equal(factor, np.full((2, 3, 3), 10*i+j+1j))


def test_validation_and_engine_exceptions_propagate(fixture, monkeypatch):
    system, basis, aux, kcart, controls, calls, refs = fixture
    for budget in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            p._build_canonical_lpq_cache(system, basis, kcart, "aux", **controls,
                                         batch_memory_bytes=budget)
    with pytest.raises(ValueError, match="one coefficient block per character"):
        p._build_canonical_lov_cache(system, basis, kcart, [], 1, "aux", **controls)
    assert not calls
    error = RuntimeError("metric failure")

    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(p, "build_lpq_bloch_native_fft_shared_q", fail)
    with pytest.raises(RuntimeError) as caught:
        p._build_canonical_lpq_cache(system, basis, kcart, "aux", **controls)
    assert caught.value is error
    monkeypatch.setattr(p, "build_lpq_bloch_native_fft_shared_q",
                        lambda *a, **kw: [np.zeros((1, 3, 3))]*3)
    with pytest.raises(RuntimeError, match="cderi rank mismatch"):
        p._build_canonical_lov_cache(system, basis, kcart, [np.eye(3)]*3, 1, "aux", **controls)
    monkeypatch.setattr(p, "build_lpq_bloch_native_fft_shared_q", lambda *a, **kw: [])
    with pytest.raises(RuntimeError, match="pair count mismatch"):
        p._build_canonical_lpq_cache(system, basis, kcart, "aux", **controls)


@pytest.mark.slow
def test_native_nonself_pair_batch_operator_and_mp2():
    import vibeqc as vq
    from vibeqc.aux_basis import default_aux_for, _build_coulomb_from_diagonal_factors
    from vibeqc.periodic.chi import scf
    from vibeqc.periodic_k_gdf import _build_k_from_lpq_cache

    system = vq.PeriodicSystem(3, 6*np.eye(3),
        [vq.Atom(1, [3., 3., 2.3]), vq.Atom(1, [3., 3., 3.7])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = (3, 1, 1)
    kp = scf.cyclic_gamma_mesh(system, mesh)
    opt = vq.PeriodicRHFOptions()
    opt.conv_tol_energy, opt.conv_tol_grad = 1e-11, 1e-9
    hf = scf.run_aiccm2026dev_b_rhf(system, basis, mesh, opt,
        backend="ri", fit_source="classic", progress=False, verbose=0)
    assert hf.converged
    C, eps = np.asarray(hf.mo_coeffs), np.asarray(hf.mo_energies)
    D = np.asarray([2*c[:, :1]@c[:, :1].conj().T for c in C])
    F, S = np.asarray(hf.fock), np.asarray(hf.overlap)
    assert np.max(np.abs(F@D@S-S@D@F)) < 1e-9
    assert np.max(np.abs(F@C-(S@C)*eps[:, None, :])) < 1e-9
    controls = dict(lattice_cutoff_bohr=15.,rsgdf_ke_cutoff=200.,gdf_linear_dep_threshold=1e-9)
    kcart = np.asarray(kp.kpoints_cart)
    name = default_aux_for(basis.name)
    base, aux = p._build_canonical_lpq_cache_pairwise(system,basis,kcart,name,**controls)
    w = np.asarray(kp.weights)

    def jk(cache):
        return (np.asarray(_build_coulomb_from_diagonal_factors([cache[i,i] for i in range(3)],D,w)),
                np.asarray(_build_k_from_lpq_cache(cache,D,w,nbasis=basis.nbasis)))

    def energy(cache):
        lov = {key:p.aiccm2026dev_b_3index_mo_transform_complex(
            np.ascontiguousarray(L),np.ascontiguousarray(C[key[0],:,:1]),
            np.ascontiguousarray(C[key[1],:,1:])) for key,L in cache.items()}
        ss, os, im = p.aiccm2026dev_b_mp2_energy_from_lov(lov,np.ascontiguousarray(eps),mesh,(0,0,0),1e-12)
        return ss+os

    J0,K0 = jk(base)
    e0 = energy(base)
    for budget in (p._CLASSIC_BATCH_MEMORY_BYTES, 3*16*aux.nbasis*basis.nbasis**2):
        batch,_ = p._build_canonical_lpq_cache(system,basis,kcart,name,**controls,batch_memory_bytes=budget)
        J,K = jk(batch)
        assert np.max(np.abs(J-J0)) < 1e-10
        assert np.max(np.abs(K-K0)) < 1e-10
        assert abs(.5*np.einsum('k,kab,kba->',w,D,J-J0-.5*(K-K0)).real) < 1e-10
        assert abs(energy(batch)-e0) < 1e-10
        lov,_ = p._build_canonical_lov_cache(system,basis,kcart,C,1,name,**controls,batch_memory_bytes=budget)
        ss,os,im = p.aiccm2026dev_b_mp2_energy_from_lov(lov,np.ascontiguousarray(eps),mesh,(0,0,0),1e-12)
        assert abs(ss+os-e0) < 1e-10
