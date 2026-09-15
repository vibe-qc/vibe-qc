"""Pure-math regression tests for the ROHF-MP2 reference kernel.

Exercises ``vibeqc.dlpno._ccsd_ref.run_ref_ump2`` / ``run_ref_rohf_mp2`` on
synthetic density-fitted integrals — no SCF, no compiled core needed beyond
importing the package. The kernel is checked against four independent
references:

* closed-shell canonical → textbook spatial RMP2;
* open-shell canonical → an independent spatial-form UMP2 (incl. OS/SS split);
* open-shell **semicanonical** (ROHF, ``f_ov ≠ 0`` → non-trivial singles) →
  the same spatial-form UMP2 reference;
* closed-shell semicanonical → the gold-standard Slater-Condon CI engine
  (``build_hamiltonian_matrix_unrestricted``) evaluated as a sum-over-states
  second-order perturbation series;

plus an exact two-level singles toy and invariance of the energy under
closed/open/virtual orbital rotations (the ROHF MP2 symmetry).

These are the build-free correctness anchors for an eventual C++ ROHF-MP2,
mirroring ``test_dlpno_ccsd.py``'s role for the spin-orbital CC kernel.
"""

import numpy as np
import pytest

from vibeqc.dlpno._ccsd_ref import run_ref_rohf_mp2, run_ref_ump2
from vibeqc.solvers._slater_condon import build_hamiltonian_matrix_unrestricted

RNG = np.random.default_rng(20260620)
TOL = 1e-9


def _random_system(n_mo, n_aux, scale=1.0):
    """Symmetric core h and a DF B-tensor; chemist eri = Σ_P B·B."""
    h = RNG.standard_normal((n_mo, n_mo))
    h = 0.5 * (h + h.T)
    B = scale * RNG.standard_normal((n_aux, n_mo, n_mo))
    B = 0.5 * (B + B.transpose(0, 2, 1))
    return h, B


def _chem(B1, B2):
    return np.einsum("Ppq,Prs->pqrs", B1, B2, optimize=True)


def _build_fock(h, B, occ_a, occ_b):
    """Genuine per-spin MO Fock for the determinant (occ_a, occ_b)."""
    V = _chem(B, B)
    n = h.shape[0]
    Da = np.zeros(n)
    Da[list(occ_a)] = 1.0
    Db = np.zeros(n)
    Db[list(occ_b)] = 1.0
    J = np.einsum("pqrs,rs->pq", V, np.diag(Da + Db))
    Ka = np.einsum("prqs,rs->pq", V, np.diag(Da))
    Kb = np.einsum("prqs,rs->pq", V, np.diag(Db))
    return h + J - Ka, h + J - Kb


def _rmp2_textbook(eps, V, no):
    """Closed-shell spatial RMP2: Σ (ia|jb)[2(ia|jb)−(ib|ja)]/Δ."""
    n = len(eps)
    iajb = V[:no, no:, :no, no:]          # (ia|jb) -> [i,a,j,b]
    ibja = iajb.transpose(0, 3, 2, 1)     # (ib|ja)
    eo, ev = eps[:no], eps[no:]
    d = (eo[:, None, None, None] - ev[None, :, None, None]
         + eo[None, None, :, None] - ev[None, None, None, :])
    return float(np.sum(iajb * (2 * iajb - ibja) / d))


def _sc(f, B, no):
    """Independent semicanonicalisation (diagonalise o-o and v-v blocks)."""
    n = f.shape[0]
    U = np.zeros((n, n))
    e = np.zeros(n)
    eo, Vo = np.linalg.eigh(f[:no, :no])
    U[:no, :no] = Vo
    e[:no] = eo
    ev, Vv = np.linalg.eigh(f[no:, no:])
    U[no:, no:] = Vv
    e[no:] = ev
    return U.T @ f @ U, np.einsum("Ppq,pi,qj->Pij", B, U, U, optimize=True), e


def _spatial_ump2_ref(fa, fb, Ba, Bb, na, nb):
    """Independent spatial-form UMP2 + semicanonical singles."""
    fa, Ba, ea = _sc(fa, Ba, na)
    fb, Bb, eb = _sc(fb, Bb, nb)
    es = np.sum(fa[:na, na:] ** 2 / (ea[:na, None] - ea[None, na:]))
    es += np.sum(fb[:nb, nb:] ** 2 / (eb[:nb, None] - eb[None, nb:]))

    def ss(V, e, no):
        A = V[:no, no:, :no, no:]
        asym = A - A.transpose(0, 3, 2, 1)
        eo, ev = e[:no], e[no:]
        d = (eo[:, None, None, None] - ev[None, :, None, None]
             + eo[None, None, :, None] - ev[None, None, None, :])
        return 0.25 * float(np.sum(asym ** 2 / d))

    e_ss = ss(_chem(Ba, Ba), ea, na) + ss(_chem(Bb, Bb), eb, nb)
    Aab = _chem(Ba, Bb)[:na, na:, :nb, nb:]
    d = (ea[:na, None, None, None] - ea[None, na:, None, None]
         + eb[None, None, :nb, None] - eb[None, None, None, nb:])
    e_os = float(np.sum(Aab ** 2 / d))
    return dict(e_singles=es, e_ss=e_ss, e_os=e_os,
                e_doubles=e_ss + e_os, e_corr=es + e_ss + e_os)


def _sc_pt2_closed(h, B, nocc):
    """Gold-standard PT2 via the Slater-Condon CI engine (closed shell).

    E2 = Σ_{n≠0} ⟨n|H|0⟩² / (E0⁰ − En⁰) over singles+doubles, with the
    zeroth-order Hamiltonian the diagonal (semicanonical) Fock.
    """
    fa, _ = _build_fock(h, B, range(nocc), range(nocc))   # closed: fa == fb
    _, B_sc, eps = _sc(fa, B, nocc)
    n = fa.shape[0]
    U = np.zeros((n, n))
    _, Vo = np.linalg.eigh(fa[:nocc, :nocc])
    U[:nocc, :nocc] = Vo
    _, Vv = np.linalg.eigh(fa[nocc:, nocc:])
    U[nocc:, nocc:] = Vv
    h_sc = U.T @ h @ U
    h2e_phys = _chem(B_sc, B_sc).transpose(0, 2, 1, 3)   # chemist -> physicist

    occ = tuple(range(nocc))
    vir = list(range(nocc, n))
    hf = (occ, occ)
    dets = {hf}

    def repl(t, outs, ins):
        s = set(t)
        for o_, i_ in zip(outs, ins):
            s.discard(o_)
            s.add(i_)
        return tuple(sorted(s))

    for i in occ:
        for a in vir:
            dets.add((repl(hf[0], [i], [a]), hf[1]))
            dets.add((hf[0], repl(hf[1], [i], [a])))
    for i in occ:
        for j in occ:
            if j <= i:
                continue
            for a in vir:
                for b in vir:
                    if b <= a:
                        continue
                    dets.add((repl(hf[0], [i, j], [a, b]), hf[1]))
                    dets.add((hf[0], repl(hf[1], [i, j], [a, b])))
    for i in occ:
        for a in vir:
            for j in occ:
                for b in vir:
                    dets.add((repl(hf[0], [i], [a]), repl(hf[1], [j], [b])))

    det_list = [hf] + sorted(dets - {hf})
    H = build_hamiltonian_matrix_unrestricted(det_list, h_sc, h2e_phys)

    def e0(det):
        return sum(eps[p] for p in det[0]) + sum(eps[p] for p in det[1])

    E0 = e0(hf)
    e2 = 0.0
    for k in range(1, len(det_list)):
        denom = E0 - e0(det_list[k])
        if abs(denom) > 1e-12:
            e2 += H[k, 0] ** 2 / denom
    return e2


def test_closed_shell_canonical_equals_textbook_rmp2():
    # Diagonal Fock (canonical) -> f_ov = 0, singles vanish, doubles == RMP2.
    _, B = _random_system(6, 9)
    eps = np.array([-2.0, -1.5, 0.5, 1.0, 1.5, 2.0])
    fc = np.diag(eps)
    res = run_ref_ump2(fc, fc, B, B, 2, 2)
    assert abs(res.e_singles) < 1e-12
    assert abs(res.e_corr - _rmp2_textbook(eps, _chem(B, B), 2)) < TOL


def test_open_shell_canonical_equals_spatial_umP2():
    h, B = _random_system(7, 11)
    fa, fb = _build_fock(h, B, range(3), range(1))
    wa, Ca = np.linalg.eigh(fa)
    wb, Cb = np.linalg.eigh(fb)
    Ba = np.einsum("Ppq,pi,qj->Pij", B, Ca, Ca, optimize=True)
    Bb = np.einsum("Ppq,pi,qj->Pij", B, Cb, Cb, optimize=True)
    fa_c, fb_c = np.diag(wa), np.diag(wb)
    res = run_ref_ump2(fa_c, fb_c, Ba, Bb, 3, 1)
    ref = _spatial_ump2_ref(fa_c, fb_c, Ba, Bb, 3, 1)
    assert abs(res.e_singles) < 1e-9          # canonical -> Brillouin
    assert abs(res.e_corr - ref["e_corr"]) < TOL
    assert abs(res.e_os - ref["e_os"]) < TOL
    assert abs(res.e_ss - ref["e_ss"]) < TOL


def test_rohf_semicanonical_equals_spatial_ref_with_singles():
    h, B = _random_system(7, 11)
    fa, fb = _build_fock(h, B, range(3), range(1))   # non-diagonal -> f_ov != 0
    res = run_ref_rohf_mp2(fa, fb, B, B, 3, 1)
    ref = _spatial_ump2_ref(fa, fb, B, B, 3, 1)
    assert abs(res.e_singles) > 1e-3                  # singles must be present
    assert abs(res.e_singles - ref["e_singles"]) < 1e-8
    assert abs(res.e_doubles - ref["e_doubles"]) < 1e-8
    assert abs(res.e_os - ref["e_os"]) < 1e-8
    assert abs(res.e_ss - ref["e_ss"]) < 1e-8
    assert abs(res.e_corr - ref["e_corr"]) < 1e-8


def test_closed_shell_semicanonical_equals_slater_condon_pt2():
    h, B = _random_system(5, 8)
    fa, fb = _build_fock(h, B, range(2), range(2))
    res = run_ref_ump2(fa, fb, B, B, 2, 2)
    assert abs(res.e_singles) > 1e-3                  # non-canonical orbitals
    assert abs(res.e_corr - _sc_pt2_closed(h, B, 2)) < 1e-8


def test_two_level_singles_toy_exact():
    f = np.array([[-0.5, 0.13], [0.13, 0.42]])
    B0 = np.zeros((1, 2, 2))                          # no two-electron -> no doubles
    res = run_ref_ump2(f, f, B0, B0, 1, 1)
    expected = 2 * f[0, 1] ** 2 / (f[0, 0] - f[1, 1])
    assert abs(res.e_doubles) < 1e-12
    assert abs(res.e_singles - expected) < TOL


def test_invariant_under_closed_open_virtual_rotation():
    from scipy.linalg import expm
    na, nb, n = 3, 1, 7
    h, B = _random_system(n, 11, scale=0.2)
    fa, fb = _build_fock(h, B, range(na), range(nb))
    for f in (fa, fb):
        f[na:, na:] += 40 * np.eye(n - na)           # gap occ below vir
    res0 = run_ref_rohf_mp2(fa, fb, B, B, na, nb)

    def rot(sz):
        A = RNG.standard_normal((sz, sz))
        A -= A.T
        return expm(A)

    Q = np.eye(n)
    Q[nb:na, nb:na] = rot(na - nb)                    # open block
    Q[na:, na:] = rot(n - na)                         # virtual block
    faR = Q.T @ fa @ Q
    fbR = Q.T @ fb @ Q
    BR = np.einsum("Ppq,pi,qj->Pij", B, Q, Q, optimize=True)
    resR = run_ref_rohf_mp2(faR, fbR, BR, BR, na, nb)
    assert abs(resR.e_corr - res0.e_corr) < 1e-8


def test_rohf_alias_matches_ump2():
    h, B = _random_system(6, 9)
    fa, fb = _build_fock(h, B, range(3), range(1))
    a = run_ref_ump2(fa, fb, B, B, 3, 1)
    b = run_ref_rohf_mp2(fa, fb, B, B, 3, 1)
    assert abs(a.e_corr - b.e_corr) < 1e-12


def test_public_rohf_mp2_allows_frozen_core_to_exhaust_beta_space(
    monkeypatch,
):
    """A valid zero-active-beta boundary is not rejected as over-frozen."""
    import vibeqc as vq
    from vibeqc.cc import run_rohf_mp2

    molecule = vq.Molecule([vq.Atom(5, [0.0, 0.0, 0.0])], multiplicity=4)
    n_mo = 6
    orbital_energies = np.array([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0])
    reference = type(
        "ROHFReference",
        (),
        {
            "converged": True,
            "mo_coeffs": np.eye(n_mo),
            "fock_alpha": np.diag(orbital_energies),
            "fock_beta": np.diag(orbital_energies + 10.0),
            "energy": -20.0,
        },
    )()
    orbital_basis = type("OrbitalBasis", (), {"name": "synthetic"})()

    class ZeroDensityFitting:
        def __init__(self, *args, **kwargs):
            pass

        def mo_transform(self, left, right):
            return np.zeros((1, left.shape[1], right.shape[1]))

    monkeypatch.setattr(vq, "BasisSet", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "vibeqc.density_fitting.DensityFitting",
        ZeroDensityFitting,
    )

    result = run_rohf_mp2(
        molecule,
        orbital_basis,
        reference,
        n_frozen_core=1,
        aux_basis="synthetic-ri",
    )

    assert result.e_corr == pytest.approx(0.0)
    assert np.isfinite(result.e_total)
