"""Excitation-case-partitioned IC-CASPT2 (roadmap 25i, M24 scaffold).

The dense determinant IC-CASPT2 (`_mrpt._ic_caspt2_solve`) is the exact
oracle: the case-partitioned engine must reproduce its second-order energy
to machine precision while organizing the FOIS into excitation-case blocks
(the structure later milestones RDM-contract).  This file pins:

* the case classifier + active-index (RDM-order) accounting;
* `caspt2_corr_cases` == the dense engine across the full case structure
  (LiH/6-31G CAS(2,4) exercises all eleven external cases, 0A..3A);
* the unsupported IPEA path raises.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import build_hamiltonian_mo, get_hf_orbital_provider
from vibeqc.solvers import _mrpt
from vibeqc.solvers._caspt2_cases import (
    _apply_e_dict,
    _ci_to_dict,
    _columns_combinatorial,
    _dot_dict,
    _overlap_sigma_raw,
    _transition_rdms123,
    active_eprod,
    apply_H0,
    apply_S,
    assemble_dense,
    case_active_count,
    caspt2_corr_cases,
    caspt2_corr_rdm,
    caspt2_corr_rdm_matfree,
    classify_case,
    eprod_expect,
    is_internal_case,
    overlap_a_i,
    overlap_case_structure,
    overlap_matrix_from_rdms,
    overlap_s_a,
    overlap_s_i,
    overlap_ss_ii,
    rdm123_any,
    rhs_from_rdms,
    zeroth_order_from_rdms,
    zeroth_order_from_rdms_f3,
)

H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)
LIH = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])


def _ham(mol, basis):
    b = BasisSet(mol, basis)
    c = get_hf_orbital_provider(mol, b, method="rhf")
    return build_hamiltonian_mo(mol, b, c)


class TestCaseClassifier:
    def test_labels_and_active_count(self):
        # n_core=2 inactive (0,1); active (2,3,4); secondary (5,...)
        nc, na, norb = 2, 3, 8
        # Ê_{5<-0}: secondary <- inactive = S<-I, zero active
        assert classify_case((5,), (0,), nc, na, norb) == ("S", "I")
        assert case_active_count(("S", "I")) == 0
        # Ê_{5,2 <- 0,3}: (S,A)<-(I,A) -> AS<-AI, two active
        case = classify_case((5, 2), (0, 3), nc, na, norb)
        assert case == ("AS", "AI")
        assert case_active_count(case) == 2
        # all-active double is internal (projected out of the FOIS)
        assert is_internal_case(classify_case((2, 3), (3, 4), nc, na, norb))
        assert not is_internal_case(("AS", "AI"))

    def test_active_count_is_rdm_order(self):
        # The overlap of a case needs an RDM of order = its active-index count.
        assert case_active_count(("SS", "II")) == 0   # MP2-like, no RDM
        assert case_active_count(("AA", "II")) == 2    # 2-RDM
        assert case_active_count(("AA", "AI")) == 3    # 3-RDM (S); 4-RDM (H0)->F3


class TestCaseScaffoldMatchesDense:
    """caspt2_corr_cases reproduces the dense IC-CASPT2 to machine precision."""

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec",
        [
            (H2, "6-31g", 0, 2, 2),
            (H2O, "sto-3g", 3, 4, 4),
            (LIH, "6-31g", 1, 4, 2),  # full 11-case structure, 0A..3A
        ],
    )
    def test_e2_matches_dense(self, mol, basis, n_core, n_act, n_elec):
        H = _ham(mol, basis)
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, n_core, n_act, n_elec, 0)
        e_dense, nb_dense = _mrpt._ic_caspt2_solve(P, n_core, n_act)
        e_case, nb_case, ranges = caspt2_corr_cases(P, n_core, n_act)
        assert abs(e_case - e_dense) < 1e-11
        assert nb_case == nb_dense  # same FOIS dimension after orthonormalization
        # every case key is a well-formed (created, annih) label pair
        for created_labels, annih_labels in ranges:
            assert set(created_labels) <= {"A", "S"}
            assert set(annih_labels) <= {"I", "A"}

    def test_lih_exercises_all_external_cases(self):
        # LiH/6-31G CAS(2,4) spans every external case from 0 to 3 active
        # indices: the full structure the RDM contraction must cover.
        H = _ham(LIH, "6-31g")
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 1, 4, 2, 0)
        _e, _nb, ranges = caspt2_corr_cases(P, 1, 4)
        active_orders = {case_active_count(c) for c in ranges}
        assert active_orders == {0, 1, 2, 3}
        # none of the retained cases are internal (those are projected out)
        assert all(not is_internal_case(c) for c in ranges)

    def test_imaginary_shift_matches_dense(self):
        H = _ham(H2O, "sto-3g")
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 3, 4, 4, 0)
        e_dense, _ = _mrpt._ic_caspt2_solve(P, 3, 4, imaginary=0.1)
        e_case, _, _ = caspt2_corr_cases(P, 3, 4, imaginary=0.1)
        assert abs(e_case - e_dense) < 1e-11

    def test_ipea_unsupported(self):
        H = _ham(H2O, "sto-3g")
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 3, 4, 4, 0)
        with pytest.raises(NotImplementedError, match="IPEA"):
            caspt2_corr_cases(P, 3, 4, ipea=0.25)


class TestCaseOverlapStructure:
    """The FOIS metric S is block-diagonal except 3 single<->double pairs.

    Discovered + validated against the dense engine (M25 scoping): the
    naive index-label cases are NOT orthogonal subspaces: a double with
    an active spectator (Ê_{a,t}Ê_{t,i}) reduces to a net single
    (Ê_{a,i}) and overlaps it.  Exactly three single<->double pairs
    couple; every other case block is independent.  This pins the
    structure the RDM contraction must respect (diagonal blocks
    independent; the three couplings handled jointly, the singles folding
    into the doubles' space as in OpenMolcas's case removal).
    """

    def test_only_single_double_redundancies_couple(self):
        H = _ham(LIH, "6-31g")  # spans all external cases incl. all 3 pairs
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 1, 4, 2, 0)
        coupled = overlap_case_structure(P, 1, 4)
        expected = {
            frozenset((("S", "I"), ("AS", "AI"))),
            frozenset((("A", "I"), ("AA", "AI"))),
            frozenset((("S", "A"), ("AS", "AA"))),
        }
        assert coupled == expected
        # every coupled pair is a single with the double that reduces to it
        # via an active spectator pair Ê_{a,t}Ê_{t,i} -> Ê_{a,i}: the double
        # carries two more active indices (the t,t spectator) than the single.
        for pair in coupled:
            orders = sorted(case_active_count(c) for c in pair)
            sizes = sorted(len(c[0]) + len(c[1]) for c in pair)
            assert sizes == [2, 4]             # a single (2 idx) + a double
            assert orders[1] == orders[0] + 2  # the spectator active pair

    def test_assemble_dense_reproduces_solve(self):
        # The extracted assemble_dense oracle, solved, equals the dense
        # engine (guards the M24 refactor and feeds M25 block validation).
        from scipy.linalg import eigh

        H = _ham(H2O, "sto-3g")
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 3, 4, 4, 0)
        b = assemble_dense(P, 3, 4)
        S, H0_raw, V_raw, E0 = b["S"], b["H0_raw"], b["V_raw"], b["E0"]
        w, U = eigh(S)
        keep = w > 1e-8
        T = U[:, keep] / np.sqrt(w[keep])
        d, Q = eigh(T.T @ H0_raw @ T - E0 * np.eye(int(keep.sum())))
        Vp = Q.T @ (T.T @ V_raw)
        e2 = float(-np.sum(Vp**2 / d))
        e_dense, _ = _mrpt._ic_caspt2_solve(P, 3, 4)
        assert abs(e2 - e_dense) < 1e-11
        # column case labels partition the contracted basis
        assert len(b["col_case"]) == S.shape[0]


class TestOverlapRDMBlocks:
    """RDM-contracted FOIS overlap closed forms (M25), validated per-element
    against the determinant primitive ⟨0|Ê†Ê|0⟩ (the dense engine's
    apply_1body on the reference) on H2O/6-31G CAS(4,4).
    """

    @staticmethod
    def _setup():
        from vibeqc.solvers import make_rdm12
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(H2O, "6-31g")
        nc, na = 3, 4
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, nc, na, 4, 0)
        norb = P["norb"]
        ref, ap, dot = P["ref"], _mrpt.apply_1body, _mrpt._dot
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=4, n_active_orb=na, n_core=nc,
                     nuclear_repulsion=0.0, ms2=0)
        rdm1, _ = make_rdm12(res.ci_coeffs, res.determinants, na)

        def Eop(state, p, q):
            e = np.zeros((norb, norb))
            e[p, q] = 1.0
            return ap(state, e, norb)

        def det_ov(t1, t2):  # ⟨E_p1q1 E_r1s1 0 | E_p2q2 E_r2s2 0⟩
            def cand(p, q, r, s):
                return Eop(Eop(ref, r, s), p, q)
            return dot(cand(*t1), cand(*t2))

        def det_single(p, q, pp, qp):
            return dot(Eop(ref, p, q), Eop(ref, pp, qp))

        inact = list(range(nc))
        sec = list(range(nc + na, norb))
        act = list(range(nc, nc + na))
        return nc, rdm1, inact, sec, act, det_ov, det_single

    def test_s_i_single(self):
        nc, rdm1, inact, sec, act, _ov, dets = self._setup()
        for a in sec:
            for i in inact:
                for ap in sec:
                    for ip in inact:
                        assert abs(overlap_s_i(a, i, ap, ip)
                                   - dets(a, i, ap, ip)) < 1e-10

    def test_s_a_single_particle_density(self):
        nc, rdm1, inact, sec, act, _ov, dets = self._setup()
        for a in sec[:2]:
            for t in act:
                for ap in sec[:2]:
                    for tp in act:
                        assert abs(overlap_s_a(a, t, ap, tp, rdm1, nc)
                                   - dets(a, t, ap, tp)) < 1e-10

    def test_a_i_single_hole_density(self):
        nc, rdm1, inact, sec, act, _ov, dets = self._setup()
        for t in act:
            for i in inact:
                for tp in act:
                    for ip in inact:
                        assert abs(overlap_a_i(t, i, tp, ip, rdm1, nc)
                                   - dets(t, i, tp, ip)) < 1e-10

    def test_ss_ii_double_closed_shell(self):
        nc, rdm1, inact, sec, act, ov, _d = self._setup()
        a, b = sec[0], sec[1]
        i, j = inact[0], inact[1]
        # full grid of bra (a,i,b,j) vs every ket permutation of the same
        # two virtual + two inactive labels, plus a few off-block tuples
        bra = (a, i, b, j)
        kets = [(a, i, b, j), (b, i, a, j), (a, j, b, i), (b, j, a, i),
                (a, i, a, j), (a, i, b, i), (sec[2], i, b, j), (a, inact[2], b, j)]
        for ket in kets:
            got = overlap_ss_ii(*bra, *ket)
            assert abs(got - ov(bra, ket)) < 1e-10


class TestActiveEProduct:
    """The generic active E-product engine ``active_eprod`` reduces an
    arbitrary product of spin-summed active generators to the active RDMs,
    validated element-wise against the determinant primitive
    ⟨0|Ê…Ê|0⟩ on H2O/6-31G CAS(4,4).  This is the M25/M26 foundation: the
    per-case overlap closed forms above are fast-path specializations of it,
    and S / V / H₀ all assemble from it once the external indices factor out.
    """

    @staticmethod
    def _setup():
        from vibeqc.solvers import make_rdm123
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(H2O, "6-31g")
        nc, na = 3, 4
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, nc, na, 4, 0)
        norb = P["norb"]
        ref, ap, dot = P["ref"], _mrpt.apply_1body, _mrpt._dot
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=4, n_active_orb=na, n_core=nc,
                     nuclear_repulsion=0.0, ms2=0)
        rdm1, dm2, dm3 = make_rdm123(res.ci_coeffs, res.determinants, na)

        def Eop(state, p, q):
            e = np.zeros((norb, norb))
            e[p, q] = 1.0
            return ap(state, e, norb)

        def det_prod(full_ops):
            # ⟨0| Ê_{full_ops[0]} … |0⟩ with FULL-orbital indices: apply the
            # generators right-to-left to |0⟩, then dot with ⟨0|.
            st = ref
            for (p, q) in reversed(full_ops):
                st = Eop(st, p, q)
            return dot(ref, st)

        return nc, na, rdm1, dm2, dm3, det_prod

    def test_k1_equals_rdm1(self):
        nc, na, rdm1, dm2, dm3, dets = self._setup()
        for p in range(na):
            for q in range(na):
                got = active_eprod([(p, q)], rdm1, dm2, dm3)
                ref = dets([(p + nc, q + nc)])
                assert abs(got - ref) < 1e-10

    def test_k2_equals_det_primitive(self):
        nc, na, rdm1, dm2, dm3, dets = self._setup()
        for p1 in range(na):
            for q1 in range(na):
                for p2 in range(na):
                    for q2 in range(na):
                        got = active_eprod(
                            [(p1, q1), (p2, q2)], rdm1, dm2, dm3)
                        ref = dets([(p1 + nc, q1 + nc), (p2 + nc, q2 + nc)])
                        assert abs(got - ref) < 1e-10

    def test_k3_equals_det_primitive(self):
        # The full n_act**6 grid (4096 triples) exercises every delta branch
        # of the 3-RDM reduction, including the δδ·γ term, off-diagonal.
        nc, na, rdm1, dm2, dm3, dets = self._setup()
        for p1 in range(na):
            for q1 in range(na):
                for p2 in range(na):
                    for q2 in range(na):
                        for p3 in range(na):
                            for q3 in range(na):
                                got = active_eprod(
                                    [(p1, q1), (p2, q2), (p3, q3)],
                                    rdm1, dm2, dm3)
                                ref = dets([(p1 + nc, q1 + nc),
                                            (p2 + nc, q2 + nc),
                                            (p3 + nc, q3 + nc)])
                                assert abs(got - ref) < 1e-10

    def test_k4_unsupported(self):
        nc, na, rdm1, dm2, dm3, _d = self._setup()
        with pytest.raises(NotImplementedError, match="F3"):
            active_eprod([(0, 0), (1, 1), (2, 2), (3, 3)], rdm1, dm2, dm3)


class TestMixedReducer:
    """The mixed-index reducer ``eprod_expect`` evaluates a FOIS overlap
    element ⟨0|Ê_I† Ê_J|0⟩ (generators with mixed I/A/S indices) by reducing
    the external indices to the active engine, validated element-wise against
    the determinant primitive on H2O/6-31G CAS(4,4).
    """

    @staticmethod
    def _setup():
        from vibeqc.solvers import make_rdm123
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(H2O, "6-31g")
        nc, na = 3, 4
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, nc, na, 4, 0)
        norb = P["norb"]
        ref, ap, dot = P["ref"], _mrpt.apply_1body, _mrpt._dot
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=4, n_active_orb=na, n_core=nc,
                     nuclear_repulsion=0.0, ms2=0)
        rdm1, dm2, dm3 = make_rdm123(res.ci_coeffs, res.determinants, na)

        def Eop(state, p, q):
            e = np.zeros((norb, norb))
            e[p, q] = 1.0
            return ap(state, e, norb)

        def det_prod(ops):
            st = ref
            for (p, q) in reversed(ops):
                st = Eop(st, p, q)
            return dot(ref, st)

        inact = list(range(nc))
        act = list(range(nc, nc + na))
        sec = list(range(nc + na, norb))
        return nc, na, norb, rdm1, dm2, dm3, inact, act, sec, det_prod

    @staticmethod
    def _dag(ops):
        return [(q, p) for (p, q) in reversed(ops)]

    def test_singles_overlap_all_external_characters(self):
        # every single<-single FOIS overlap ⟨0|Ê_qp Ê_rs|0⟩ over the full
        # virtual x occupied excitation set (0- and 1-active singles).
        nc, na, norb, r1, d2, d3, inact, act, sec, dets = self._setup()
        virt, occ = act + sec, inact + act
        exc = [(p, q) for p in virt for q in occ if p != q]
        for (p, q) in exc:
            for (r, s) in exc:
                ops = self._dag([(p, q)]) + [(r, s)]
                got = eprod_expect(ops, nc, na, r1, d2, d3)
                assert abs(got - dets(ops)) < 1e-9

    def test_retained_doubles_overlap_sample(self):
        # a structured sample of double<-double overlaps spanning the 1-/2-/
        # 3-active retained cases; internal (all-active) products are skipped
        # (projected out of the FOIS, they alone would need the 4-RDM).
        nc, na, norb, r1, d2, d3, inact, act, sec, dets = self._setup()
        virt, occ = act + sec, inact + act
        a, bb = sec[0], sec[1]
        t, u = act[0], act[1]
        i, j = inact[0], inact[1]
        dbl = [
            [(a, i), (bb, j)],          # SS<-II
            [(a, t), (bb, j)],          # SS<-AI
            [(a, t), (bb, u)],          # SS<-AA
            [(a, i), (t, j)],           # AS<-II
            [(a, t), (u, i)],           # AS<-AI
            [(a, t), (u, j)],           # AS<-AI / AS<-AA mix
            [(t, i), (u, j)],           # AA<-II
            [(t, i), (a, u)],           # AS<-AI
            [(a, u), (t, i)],           # AS<-AI (reordered)
        ]
        for bra in dbl:
            for ket in dbl:
                ops = self._dag(bra) + ket
                if all(nc <= p < nc + na and nc <= q < nc + na
                       for (p, q) in ops):
                    continue  # internal, projected out
                got = eprod_expect(ops, nc, na, r1, d2, d3)
                assert abs(got - dets(ops)) < 1e-9


class TestOverlapFromRDMs:
    """``overlap_matrix_from_rdms`` reproduces the dense FOIS overlap ``S``
    purely from the active RDMs (the M25 deliverable (S is an RDM
    contraction, no determinant FOIS)).
    """

    @staticmethod
    def _rdms(P, n_core, n_act, n_elec):
        from vibeqc.solvers import make_rdm123
        from vibeqc.solvers._casci import casci as _casci

        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=n_elec, n_active_orb=n_act, n_core=n_core,
                     nuclear_repulsion=0.0, ms2=0)
        return make_rdm123(res.ci_coeffs, res.determinants, n_act)

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec",
        [
            (LIH, "6-31g", 1, 4, 2),   # full 11-case structure + 3 couplings
            (H2O, "sto-3g", 3, 4, 4),
            (H2, "6-31g", 0, 2, 2),
        ],
    )
    def test_S_matches_dense(self, mol, basis, n_core, n_act, n_elec):
        H = _ham(mol, basis)
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, n_core, n_act, n_elec, 0)
        rdm1, dm2, dm3 = self._rdms(P, n_core, n_act, n_elec)
        S_dense = assemble_dense(P, n_core, n_act)["S"]
        out = overlap_matrix_from_rdms(P, n_core, n_act, rdm1, dm2, dm3)
        assert out["S"].shape == S_dense.shape
        assert np.max(np.abs(out["S"] - S_dense)) < 1e-9

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec",
        [
            (H2, "6-31g", 0, 2, 2),
            (LIH, "sto-3g", 1, 2, 2),   # inactive + secondary, small
        ],
    )
    def test_V_matches_dense(self, mol, basis, n_core, n_act, n_elec):
        # The RHS V_I = ⟨0|Ê_I† Ĥ|0⟩ from the RDMs reproduces the dense
        # determinant-built V_raw.  Reducing Ĥ's all-active two-body terms
        # against an external bra returns zero through the boundary actions,
        # so no V element reaches the 4-RDM either.
        H = _ham(mol, basis)
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, n_core, n_act, n_elec, 0)
        rdm1, dm2, dm3 = self._rdms(P, n_core, n_act, n_elec)
        V_dense = assemble_dense(P, n_core, n_act)["V_raw"]
        out = rhs_from_rdms(P, n_core, n_act, rdm1, dm2, dm3)
        assert out["V"].shape == V_dense.shape
        assert np.max(np.abs(out["V"] - V_dense)) < 1e-9


class TestZerothOrderFromRDMs:
    """``zeroth_order_from_rdms`` reproduces the dense FOIS ``H0_raw`` from the
    RDMs.  Unlike S and V, H0's 3-active cases genuinely reach the 4-RDM, so
    this is also the test that pins what the F3 intermediate (M26) reproduces.
    """

    @staticmethod
    def _rdms4(P, n_core, n_act, n_elec):
        from vibeqc.solvers import make_rdm123
        from vibeqc.solvers._casci import casci as _casci
        from vibeqc.solvers._rdm import _generator_products, _index_map

        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=n_elec, n_active_orb=n_act, n_core=n_core,
                     nuclear_repulsion=0.0, ms2=0)
        rdm1, dm2, dm3 = make_rdm123(res.ci_coeffs, res.determinants, n_act)
        idx = _index_map(res.determinants)
        # raw 4-generator product (small-space validation only)
        _r1, _T2, _T3, T4 = _generator_products(
            res.ci_coeffs, res.determinants, n_act, idx, want3=True, want4=True
        )
        return rdm1, dm2, dm3, T4

    def test_H0_matches_dense(self):
        # H2/6-31g CAS(2,2) has no inactive orbitals but does span the
        # 3-active AS<-AA case, so H0 here exercises the 4-RDM path.
        H = _ham(H2, "6-31g")
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 0, 2, 2, 0)
        rdm1, dm2, dm3, T4 = self._rdms4(P, 0, 2, 2)
        H0_dense = assemble_dense(P, 0, 2)["H0_raw"]
        out = zeroth_order_from_rdms(P, 0, 2, rdm1, dm2, dm3, T4)
        assert out["H0"].shape == H0_dense.shape
        assert np.max(np.abs(out["H0"] - H0_dense)) < 1e-8

    def test_H0_needs_the_4rdm(self):
        # Without t4, H0 raises on the 3-active case: this is exactly the
        # 4-RDM that S and V never hit, and that F3 (M26) removes.
        H = _ham(H2, "6-31g")
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 0, 2, 2, 0)
        rdm1, dm2, dm3, _T4 = self._rdms4(P, 0, 2, 2)
        with pytest.raises(NotImplementedError, match="4-RDM"):
            zeroth_order_from_rdms(P, 0, 2, rdm1, dm2, dm3, t4=None)

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec",
        [
            (H2, "6-31g", 0, 2, 2),      # 3-active AS<-AA, no inactive
            (LIH, "sto-3g", 1, 2, 2),    # inactive + secondary (the s0 path)
        ],
    )
    def test_H0_f3_matches_dense_without_4rdm(self, mol, basis,
                                              n_core, n_act, n_elec):
        # The F3 / transition-RDM route reproduces H0 using only 1-/2-/3-RDMs
        # (reference + transition); the O(n_act**8) 4-RDM is never built.
        from vibeqc.solvers import make_rdm123
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(mol, basis)
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, n_core, n_act, n_elec, 0)
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=n_elec, n_active_orb=n_act, n_core=n_core,
                     nuclear_repulsion=0.0, ms2=0)
        rdm1, dm2, dm3 = make_rdm123(res.ci_coeffs, res.determinants, n_act)
        H0_dense = assemble_dense(P, n_core, n_act)["H0_raw"]
        out = zeroth_order_from_rdms_f3(
            P, n_core, n_act, rdm1, dm2, dm3,
            res.ci_coeffs, res.determinants,
        )
        assert out["H0"].shape == H0_dense.shape
        assert np.max(np.abs(out["H0"] - H0_dense)) < 1e-8


class TestCASPT2EnergyFromRDMs:
    """The full case-partitioned CASPT2 second-order energy assembled entirely
    from the RDMs (S + V + F3 H0) equals the dense OpenMolcas-pinned engine
    ``_ic_caspt2_solve`` to machine precision.  This is the M27 capstone: a
    CASPT2 correlation energy with no determinant FOIS and no 4-RDM.
    """

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec",
        [
            (H2, "6-31g", 0, 2, 2),
            (LIH, "sto-3g", 1, 2, 2),   # inactive + secondary, small
        ],
    )
    def test_corr_matches_dense_engine(self, mol, basis,
                                       n_core, n_act, n_elec):
        # Default (determinant_free=True): the FOIS columns are enumerated
        # combinatorially, so the whole energy is independent of the reference
        # determinant count.
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(mol, basis)
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, n_core, n_act, n_elec, 0)
        e_dense, nb_dense = _mrpt._ic_caspt2_solve(P, n_core, n_act)
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=n_elec, n_active_orb=n_act, n_core=n_core,
                     nuclear_repulsion=0.0, ms2=0)
        e_rdm, nb_rdm = caspt2_corr_rdm(
            P, n_core, n_act, res.ci_coeffs, res.determinants)
        assert nb_rdm == nb_dense
        assert abs(e_rdm - e_dense) < 1e-9

    def test_determinant_free_equals_build_fois(self):
        # The combinatorial (determinant-free) column enumeration spans the
        # same FOIS as the dense build_fois_by_case path: same energy + same
        # orthonormalized dimension.
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(LIH, "sto-3g")
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 1, 2, 2, 0)
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=2, n_active_orb=2, n_core=1,
                     nuclear_repulsion=0.0, ms2=0)
        e_df, nb_df = caspt2_corr_rdm(
            P, 1, 2, res.ci_coeffs, res.determinants, determinant_free=True)
        e_bf, nb_bf = caspt2_corr_rdm(
            P, 1, 2, res.ci_coeffs, res.determinants, determinant_free=False)
        assert nb_df == nb_bf
        assert abs(e_df - e_bf) < 1e-12

    def test_imaginary_shift_matches_dense(self):
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(LIH, "sto-3g")
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 1, 2, 2, 0)
        e_dense, _ = _mrpt._ic_caspt2_solve(P, 1, 2, imaginary=0.1)
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=2, n_active_orb=2, n_core=1,
                     nuclear_repulsion=0.0, ms2=0)
        e_rdm, _ = caspt2_corr_rdm(
            P, 1, 2, res.ci_coeffs, res.determinants, imaginary=0.1)
        assert abs(e_rdm - e_dense) < 1e-9


class TestSelectedReferenceRDM:
    """The RDM CASPT2 composes with a *selected* (truncated) reference, the
    route to active spaces beyond full CI.  ``rdm123_any`` builds the 3-RDM of
    any determinant list, and ``caspt2_corr_rdm`` on a truncated reference
    matches the dense selected-reference engine (``_semicanonical_prep(
    reference=…)`` + ``_ic_caspt2_solve``, the M22 path).
    """

    def test_rdm123_any_matches_make_rdm123_full_cas(self):
        from vibeqc.solvers import make_rdm123
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(H2O, "6-31g")
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, 3, 4, 4, 0)
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=4, n_active_orb=4, n_core=3,
                     nuclear_repulsion=0.0, ms2=0)
        r1a, d2a, d3a = make_rdm123(res.ci_coeffs, res.determinants, 4)
        r1b, d2b, d3b = rdm123_any(res.ci_coeffs, res.determinants, 4)
        assert np.max(np.abs(r1a - r1b)) < 1e-12
        assert np.max(np.abs(d2a - d2b)) < 1e-12
        assert np.max(np.abs(d3a - d3b)) < 1e-12

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec,keep_frac",
        [
            (LIH, "sto-3g", 1, 2, 2, 0.6),   # inactive + secondary, small
        ],
    )
    def test_selected_reference_matches_dense(self, mol, basis, n_core,
                                              n_act, n_elec, keep_frac):
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(mol, basis)
        P0 = _mrpt._semicanonical_prep(H.h1e, H.h2e, n_core, n_act, n_elec, 0)
        r = _casci(P0["h1"], P0["eri"].transpose(0, 2, 1, 3),
                   n_active_elec=n_elec, n_active_orb=n_act, n_core=n_core,
                   nuclear_repulsion=0.0, ms2=0)
        ci = np.asarray(r.ci_coeffs, dtype=float)
        # truncate to the largest-weight determinants -> a selected reference
        order = np.argsort(-np.abs(ci))
        nkeep = max(2, int(len(ci) * keep_frac))
        keep = sorted(order[:nkeep])
        sel_dets = [r.determinants[i] for i in keep]
        sel_ci = ci[keep].copy()
        sel_ci /= np.linalg.norm(sel_ci)
        assert len(sel_dets) < len(ci)  # genuinely truncated

        Psel = _mrpt._semicanonical_prep(
            H.h1e, H.h2e, n_core, n_act, n_elec, 0,
            reference=(sel_ci, sel_dets))
        e_dense, nb_dense = _mrpt._ic_caspt2_solve(Psel, n_core, n_act)
        rdms = rdm123_any(sel_ci, sel_dets, n_act)
        e_rdm, nb_rdm = caspt2_corr_rdm(
            Psel, n_core, n_act, sel_ci, sel_dets, rdms=rdms)
        assert nb_rdm == nb_dense
        assert abs(e_rdm - e_dense) < 1e-9


class TestMatrixFreeSigma:
    """The matrix-free overlap application ``apply_S`` reproduces the full
    ``S @ x`` without materializing ``S`` (M27b sigma foundation), exploiting
    that S couples only the 3 single<->double pairs.
    """

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec",
        [
            (H2, "6-31g", 0, 2, 2),
            (LIH, "6-31g", 1, 4, 2),   # all 11 cases + 3 couplings
            (H2O, "sto-3g", 3, 4, 4),
        ],
    )
    def test_apply_S_matches_full(self, mol, basis, n_core, n_act, n_elec):
        from vibeqc.solvers import make_rdm123
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(mol, basis)
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, n_core, n_act, n_elec, 0)
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=n_elec, n_active_orb=n_act, n_core=n_core,
                     nuclear_repulsion=0.0, ms2=0)
        rdm1, dm2, dm3 = make_rdm123(res.ci_coeffs, res.determinants, n_act)
        cols = _columns_combinatorial(P, n_core, n_act, rdm1, dm2, dm3)
        Sfull = overlap_matrix_from_rdms(
            P, n_core, n_act, rdm1, dm2, dm3, cols=cols)["S"]
        rng = np.random.default_rng(0)
        for _ in range(3):
            x = rng.standard_normal(Sfull.shape[0])
            sx = apply_S(x, cols, n_core, n_act, rdm1, dm2, dm3)
            assert np.max(np.abs(sx - Sfull @ x)) < 1e-9

    def test_overlap_sigma_transition(self):
        # The same block-structured overlap sigma computes the H0 *transition*
        # term <0|E_I^dag E_J|Phi> (|Phi> = Fhat_act|0>) when fed the
        # transition RDMs + s0 = <0|Phi>: same operator structure as S, hence
        # the same 3-coupling block sparsity. This is H0's term A (M27b).
        from vibeqc.solvers import make_rdm123
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(LIH, "6-31g")
        nc, na = 1, 4
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, nc, na, 2, 0)
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=2, n_active_orb=na, n_core=nc,
                     nuclear_repulsion=0.0, ms2=0)
        rdm1, dm2, dm3 = make_rdm123(res.ci_coeffs, res.determinants, na)
        cols = _columns_combinatorial(P, nc, na, rdm1, dm2, dm3)
        col_ops = cols[0]
        # |Phi> = Fhat_act|0> + transition RDMs + s0
        F = P["F"]
        bra = _ci_to_dict(res.ci_coeffs, res.determinants)
        phi: dict = {}
        for t in range(na):
            for u in range(na):
                f = F[nc + t, nc + u]
                if abs(f) > 1e-14:
                    for k, v in _apply_e_dict(t, u, bra).items():
                        phi[k] = phi.get(k, 0.0) + f * v
        s0 = _dot_dict(bra, phi)
        tr1, tdm2, tdm3 = _transition_rdms123(bra, phi, na)
        # full raw transition-overlap matrix (per element) as the oracle

        def dag(ops):
            return [(q, p) for (p, q) in reversed(ops)]

        n = len(col_ops)
        rawT = np.zeros((n, n))
        for i in range(n):
            bi = dag(col_ops[i])
            for j in range(n):
                rawT[i, j] = eprod_expect(
                    bi + col_ops[j], nc, na, tr1, tdm2, tdm3, s0=s0)
        rng = np.random.default_rng(1)
        for _ in range(3):
            y = rng.standard_normal(n)
            sig = _overlap_sigma_raw(y, cols, nc, na, tr1, tdm2, tdm3, s0=s0)
            assert np.max(np.abs(sig - rawT @ y)) < 1e-9

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec",
        [
            (H2, "6-31g", 0, 2, 2),
            (LIH, "sto-3g", 1, 2, 2),   # inactive + secondary
        ],
    )
    def test_apply_H0_matches_full(self, mol, basis, n_core, n_act, n_elec):
        from vibeqc.solvers import make_rdm123
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(mol, basis)
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, n_core, n_act, n_elec, 0)
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=n_elec, n_active_orb=n_act, n_core=n_core,
                     nuclear_repulsion=0.0, ms2=0)
        ci, dets = res.ci_coeffs, res.determinants
        rdm1, dm2, dm3 = make_rdm123(ci, dets, n_act)
        cols = _columns_combinatorial(P, n_core, n_act, rdm1, dm2, dm3)
        H0full = zeroth_order_from_rdms_f3(
            P, n_core, n_act, rdm1, dm2, dm3, ci, dets, cols=cols)["H0"]
        rng = np.random.default_rng(0)
        for _ in range(3):
            x = rng.standard_normal(H0full.shape[0])
            hx = apply_H0(x, cols, P, n_core, n_act, rdm1, dm2, dm3, ci, dets)
            assert np.max(np.abs(hx - H0full @ x)) < 1e-8


class TestMatrixFreeEnergy:
    """The fully matrix-free CASPT2 energy (MINRES on the sigma products)
    reproduces the assembled-matrix RDM energy == the dense engine (M27b).
    """

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec",
        [
            (H2, "6-31g", 0, 2, 2),
            (LIH, "sto-3g", 1, 2, 2),   # inactive + secondary
        ],
    )
    def test_matfree_matches_assembled(self, mol, basis,
                                       n_core, n_act, n_elec):
        from vibeqc.solvers._casci import casci as _casci

        H = _ham(mol, basis)
        P = _mrpt._semicanonical_prep(H.h1e, H.h2e, n_core, n_act, n_elec, 0)
        e_dense, _ = _mrpt._ic_caspt2_solve(P, n_core, n_act)
        res = _casci(P["h1"], P["eri"].transpose(0, 2, 1, 3),
                     n_active_elec=n_elec, n_active_orb=n_act, n_core=n_core,
                     nuclear_repulsion=0.0, ms2=0)
        e_mf, n_mf = caspt2_corr_rdm_matfree(
            P, n_core, n_act, res.ci_coeffs, res.determinants)
        assert abs(e_mf - e_dense) < 1e-8


class TestPublicEngineCases:
    """The case-partitioned matrix-free RDM CASPT2 is reachable from the public
    ``caspt2(engine="cases")`` and reproduces the default engine.
    """

    @pytest.mark.parametrize(
        "mol,basis,n_core,n_act,n_elec",
        [
            (H2, "6-31g", 0, 2, 2),
            (LIH, "sto-3g", 1, 2, 2),
        ],
    )
    def test_engine_cases_matches_auto(self, mol, basis, n_core, n_act, n_elec):
        from vibeqc.solvers._casci import casci as _casci
        from vibeqc.solvers._mrpt import caspt2

        H = _ham(mol, basis)
        norb = H.h1e.shape[0]
        n_virt = norb - n_core - n_act
        res = _casci(H.h1e, H.h2e, n_active_elec=n_elec, n_active_orb=n_act,
                     n_core=n_core, nuclear_repulsion=0.0, ms2=0)
        auto = caspt2(res, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
                      engine="auto")
        cases = caspt2(res, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
                       engine="cases")
        assert abs(auto.e_corr - cases.e_corr) < 1e-8

    def test_engine_cases_rejects_shift_and_ipea(self):
        from vibeqc.solvers._casci import casci as _casci
        from vibeqc.solvers._mrpt import caspt2

        H = _ham(H2, "6-31g")
        res = _casci(H.h1e, H.h2e, n_active_elec=2, n_active_orb=2, n_core=0,
                     nuclear_repulsion=0.0, ms2=0)
        with pytest.raises(ValueError, match="unshifted"):
            caspt2(res, H.h1e, H.h2e, n_core=0, n_virt=2,
                   engine="cases", imaginary=0.1)
        with pytest.raises(ValueError, match="engine"):
            caspt2(res, H.h1e, H.h2e, n_core=0, n_virt=2, engine="bogus")
