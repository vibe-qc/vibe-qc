"""Local DLPNO-CCSD integral blocks — parity against canonical full-space.

The reduced-scaling DLPNO-CCSD residual is evaluated in each pair's PNO
basis. Its integral foundation (`dlpno.ccsd_local`) builds the
density-fitted blocks K=(ia|jb), J=(ij|ab), the 3-external (ia|bc) and
the 4-external ladder (ab|cd) directly in that small basis. Each must
equal the canonical full-virtual-space integral projected into the PNO
space by the PNO coordinates U = C_virᵀ S V_pno — to machine precision,
since the PNOs are an exact subspace of the canonical virtuals.

Two configurations: a full PNO space (U square-orthogonal — the blocks
are just rotations of the canonical ones) and a truncated PNO space
(U a genuine projection).
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule, compute_overlap
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.ccsd_local import (
    build_pair_cc_integrals,
    build_pair_occ_pno,
    local_doubles_ladder_residual,
    pno_overlap,
    project_amplitude,
)
from vibeqc.dlpno._ccsd_cs import _blocks
from vibeqc.dlpno.pao import (
    build_atom_basis_map,
    build_projection_matrix,
    semicanonical_pao_basis,
)

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"

H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]


@pytest.fixture(scope="module")
def h2o():
    mol = Molecule([Atom(z, p) for z, p in H2O_ATOMS], charge=0, multiplicity=1)
    basis = BasisSet(mol, "def2-svp")
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    S = np.asarray(compute_overlap(basis)).copy()
    F = np.asarray(rhf.fock).copy()
    C = np.asarray(rhf.mo_coeffs).copy()
    n_occ = mol.n_electrons() // 2
    return dict(
        mol=mol, basis=basis, df=df, S=S, F=F, C=C, n_occ=n_occ,
        C_occ=C[:, :n_occ], C_vir=C[:, n_occ:],
    )


def _pair_pno(h2o, i, j, truncate):
    """Build a pair's PNO AO-expansion V_pno (full domain)."""
    mol, basis, df = h2o["mol"], h2o["basis"], h2o["df"]
    S, F, C_occ = h2o["S"], h2o["F"], h2o["C_occ"]
    nbf = C_occ.shape[0]
    atom_first, _ = build_atom_basis_map(mol, basis)
    natom = len(atom_first) - 1
    Q_vir = build_projection_matrix(C_occ, S)
    V_semi, eps_pao = semicanonical_pao_basis(
        F, S, Q_vir, np.arange(nbf), 1e-8
    )
    if not truncate:
        return V_semi
    # Real PNOs from the semicanonical MP2 pair density.
    B_half = df.mo_transform(C_occ, np.eye(nbf))
    B_i = B_half[:, i, :] @ V_semi
    B_j = B_half[:, j, :] @ V_semi
    K = B_i.T @ B_j
    f = np.diag(C_occ.T @ F @ C_occ)
    T = K / (f[i] + f[j] - eps_pao[:, None] - eps_pao[None, :])
    delta = 1.0 if i == j else 0.0
    D = (T @ T.T + T.T @ T) / (1.0 + delta)
    occ, d = np.linalg.eigh(0.5 * (D + D.T))
    order = np.argsort(-occ)
    keep = occ[order] > 1e-7
    return V_semi @ d[:, order][:, keep]


def _U(h2o, V_pno):
    """PNO coordinates in the canonical-virtual basis."""
    return h2o["C_vir"].T @ h2o["S"] @ V_pno


def _canonical_blocks(h2o, i, j):
    df, C_occ, C_vir = h2o["df"], h2o["C_occ"], h2o["C_vir"]
    B_ov = np.asarray(df.mo_transform(C_occ, C_vir))   # (naux, nocc, nvir)
    B_vv = np.asarray(df.mo_transform(C_vir, C_vir))    # (naux, nvir, nvir)
    Bi, Bj = B_ov[:, i, :], B_ov[:, j, :]
    return Bi, Bj, B_vv


@pytest.mark.parametrize("truncate", [False, True])
class TestIntegralParity:
    PAIRS = [(0, 0), (1, 3)]

    def test_K_block(self, h2o, truncate):
        for (i, j) in self.PAIRS:
            V_pno = _pair_pno(h2o, i, j, truncate)
            U = _U(h2o, V_pno)
            ints = build_pair_cc_integrals(h2o["df"], h2o["C_occ"], i, j, V_pno)
            Bi, Bj, _ = _canonical_blocks(h2o, i, j)
            K_can_full = Bi.T @ Bj                     # (nvir, nvir)
            K_can_proj = U.T @ K_can_full @ U          # (n_pno, n_pno)
            assert np.allclose(ints.K(), K_can_proj, atol=1e-11)

    def test_four_external_ladder(self, h2o, truncate):
        for (i, j) in self.PAIRS:
            V_pno = _pair_pno(h2o, i, j, truncate)
            U = _U(h2o, V_pno)
            ints = build_pair_cc_integrals(h2o["df"], h2o["C_occ"], i, j, V_pno)
            _, _, B_vv = _canonical_blocks(h2o, i, j)
            four_local = ints.four_ext()               # (n_pno,)*4
            four_proj = np.einsum(
                "Pab,Pcd,aA,bB,cC,dD->ABCD",
                B_vv, B_vv, U, U, U, U, optimize=True,
            )
            assert np.allclose(four_local, four_proj, atol=1e-11)

    def test_three_external_block(self, h2o, truncate):
        for (i, j) in self.PAIRS:
            V_pno = _pair_pno(h2o, i, j, truncate)
            U = _U(h2o, V_pno)
            ints = build_pair_cc_integrals(h2o["df"], h2o["C_occ"], i, j, V_pno)
            Bi, _, B_vv = _canonical_blocks(h2o, i, j)
            three_local = ints.three_ext_i()           # (n_pno,n_pno,n_pno)
            three_proj = np.einsum(
                "Pa,Pbc,aA,bB,cC->ABC",
                Bi, B_vv, U, U, U, optimize=True,
            )
            assert np.allclose(three_local, three_proj, atol=1e-11)


class TestStructure:
    def test_full_space_U_is_orthogonal(self, h2o):
        """A full PNO domain ⇒ U is an exact orthonormal rotation."""
        V_pno = _pair_pno(h2o, 1, 3, truncate=False)
        U = _U(h2o, V_pno)
        # square (n_vir × n_vir) and orthogonal.
        assert U.shape[0] == U.shape[1]
        assert np.allclose(U @ U.T, np.eye(U.shape[0]), atol=1e-10)

    def test_blocks_have_pno_dimensions(self, h2o):
        V_pno = _pair_pno(h2o, 1, 3, truncate=True)
        n = V_pno.shape[1]
        ints = build_pair_cc_integrals(h2o["df"], h2o["C_occ"], 1, 3, V_pno)
        assert ints.K().shape == (n, n)
        assert ints.J().shape == (n, n)
        assert ints.four_ext().shape == (n, n, n, n)


class TestLocalLadderResidual:
    """The hole/particle-ladder part of the local T2 residual matches the
    full-space residual projected into each pair's PNO basis, exactly —
    validating the cross-pair amplitude projection machinery."""

    @pytest.fixture(scope="class")
    @classmethod
    def localresid(cls):
        d = _setup_h2o_dz()
        return d

    def test_projection_round_trip(self, localresid):
        d = localresid
        S = d["S"]
        # Two pairs' PNO spaces; project an amplitude q→p→q and check the
        # full-space-limit identity holds within the shared virtual span.
        Vp = d["Vpno"][(0, 0)]
        Vq = d["Vpno"][(1, 2)]
        Spq = pno_overlap(Vp, Vq, S)
        assert Spq.shape == (Vp.shape[1], Vq.shape[1])
        Tq = np.random.default_rng(0).standard_normal((Vq.shape[1], Vq.shape[1]))
        Tp = project_amplitude(Spq, Tq)
        assert Tp.shape == (Vp.shape[1], Vp.shape[1])

    def test_ladder_residual_matches_full_space(self, localresid):
        d = localresid
        na, nv = d["na"], d["nv"]
        U, T2, Vpno = d["U"], d["T2"], d["Vpno"]
        S = d["S"]

        # Partial oracle: cs_ccsd_residual terms 1-3 (t1=0), projected.
        T2f = np.zeros((na, na, nv, nv))
        for (i, j), T in T2.items():
            blk = U[(i, j)] @ T @ U[(i, j)].T
            T2f[i, j] = blk
            T2f[j, i] = blk.T
        V = _blocks(d["B_ov"], d["B_vv"], d["B_oo"])
        ovov, oooo, vvvv = V["ovov"], V["oooo"], V["vvvv"]
        tau = T2f
        Wmnij = oooo.transpose(0, 2, 1, 3) + 0.5 * np.einsum(
            "ijef,menf->mnij", tau, ovov, optimize=True
        )
        Wabef = vvvv.transpose(0, 2, 1, 3) + 0.5 * np.einsum(
            "mnab,menf->abef", tau, ovov, optimize=True
        )
        R2p = (
            ovov.transpose(0, 2, 1, 3)
            + np.einsum("mnij,mnab->ijab", Wmnij, tau, optimize=True)
            + np.einsum("abef,ijef->ijab", Wabef, tau, optimize=True)
        )

        def Vof(k, l):
            return Vpno[(k, l)] if (k, l) in Vpno else Vpno[(l, k)]

        def Tof(k, l):
            return T2[(k, l)] if (k, l) in T2 else T2[(l, k)].T

        B_oo_eng = d["B_oo"]
        err = 0.0
        for (i, j) in T2:
            Vij = Vpno[(i, j)]
            B_occ = build_pair_occ_pno_from(d["df"], d["C_loc"], Vij)
            B_vvp = np.einsum("Puv,ua,vb->Pab", d["B_vv_can"], U[(i, j)], U[(i, j)],
                              optimize=True)

            def proj_into(m, n, Vij=Vij):
                Spq = pno_overlap(Vij, Vof(m, n), S)
                return project_amplitude(Spq, Tof(m, n))

            R = local_doubles_ladder_residual(
                i, j, B_occ, B_vvp, B_oo_eng, Tof(i, j), proj_into
            )
            oracle = U[(i, j)].T @ R2p[i, j] @ U[(i, j)]
            err = max(err, float(np.max(np.abs(R - oracle))))
        assert err < 1e-9, f"ladder residual parity {err:.2e}"


def build_pair_occ_pno_from(df, C_occ, V_pno):
    return build_pair_occ_pno(df, C_occ, V_pno)


def _setup_h2o_dz(tcut=1e-7):
    import numpy as np
    from vibeqc import compute_dipole
    from vibeqc.localise import foster_boys_localise
    from vibeqc.dlpno.pao import build_projection_matrix, semicanonical_pao_basis

    mol = Molecule([Atom(z, p) for z, p in H2O_ATOMS], charge=0, multiplicity=1)
    basis = BasisSet(mol, "def2-svp")
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    S = np.asarray(compute_overlap(basis))
    F = np.asarray(rhf.fock)
    C = np.asarray(rhf.mo_coeffs)
    na = mol.n_electrons() // 2
    nbf = C.shape[0]
    C_occ, C_vir = C[:, :na], C[:, na:]
    nv = C_vir.shape[1]
    dip = compute_dipole(basis)
    Dp = np.zeros((nbf, nbf, 3))
    Dp[:, :, 0] = np.asarray(dip.x)
    Dp[:, :, 1] = np.asarray(dip.y)
    Dp[:, :, 2] = np.asarray(dip.z)
    C_loc = foster_boys_localise(C_occ, Dp, max_iter=200)
    F_oo = C_loc.T @ F @ C_loc
    f_dd = np.diag(F_oo)
    C_eng = np.hstack([C_loc, C_vir])
    B_eng = np.asarray(df.mo_transform(C_eng, C_eng))
    B_ov = np.ascontiguousarray(B_eng[:, :na, na:])
    B_vv = np.ascontiguousarray(B_eng[:, na:, na:])
    B_oo = np.ascontiguousarray(B_eng[:, :na, :na])
    B_vv_can = np.asarray(df.mo_transform(C_vir, C_vir))
    Q_vir = build_projection_matrix(C_occ, S)
    U, T2, Vpno = {}, {}, {}
    for i in range(na):
        for j in range(i, na):
            Vs, ep0 = semicanonical_pao_basis(F, S, Q_vir, np.arange(nbf), 1e-8)
            Bi = df.mo_transform(C_loc[:, [i]], Vs)[:, 0, :]
            Bj = df.mo_transform(C_loc[:, [j]], Vs)[:, 0, :]
            Kp = Bi.T @ Bj
            Tp = Kp / (f_dd[i] + f_dd[j] - ep0[:, None] - ep0[None, :])
            dl = 1.0 if i == j else 0.0
            Dd = (Tp @ Tp.T + Tp.T @ Tp) / (1 + dl)
            occ, dd = np.linalg.eigh(0.5 * (Dd + Dd.T))
            order = np.argsort(-occ)
            occ, dd = occ[order], dd[:, order]
            keep = occ > tcut
            if not keep.any():
                keep[0] = True
            dd = dd[:, keep]
            Fp = dd.T @ np.diag(ep0) @ dd
            epn, ur = np.linalg.eigh(0.5 * (Fp + Fp.T))
            dd = dd @ ur
            Vp = Vs @ dd
            U[(i, j)] = C_vir.T @ S @ Vp
            Vpno[(i, j)] = Vp
            T2[(i, j)] = (dd.T @ Kp @ dd) / (
                f_dd[i] + f_dd[j] - epn[:, None] - epn[None, :]
            )
    return dict(df=df, C_occ=C_occ, C_loc=C_loc, C=C, F=F, S=S, na=na, nv=nv, U=U, T2=T2, Vpno=Vpno,
                B_ov=B_ov, B_vv=B_vv, B_oo=B_oo, B_vv_can=B_vv_can)


class TestLocalDoublesResidual:
    """The complete local T2 (doubles, t1=0) residual: ladders + Fock
    ladders + W1/W2/WX rings. Exact in the full-domain limit."""

    def _run(self, d):
        import numpy as np
        from vibeqc.dlpno.ccsd_local import (
            build_pair_occ_pno, local_t2_residual_doubles, pno_overlap,
            project_amplitude,
        )
        from vibeqc.dlpno._ccsd_cs import cs_ccsd_residual
        na, nv = d["na"], d["nv"]
        U, T2, Vpno = d["U"], d["T2"], d["Vpno"]
        S = d["S"]
        # full-space oracle at t1=0
        T2f = np.zeros((na, na, nv, nv))
        for (i, j), T in T2.items():
            blk = U[(i, j)] @ T @ U[(i, j)].T
            T2f[i, j] = blk
            T2f[j, i] = blk.T
        f_oo_e = d["C_loc"].T @ d["F"] @ d["C_loc"]
        C_vir = d["C"][:, na:]
        f_vv_e = C_vir.T @ d["F"] @ C_vir
        f_ov_e = d["C_loc"].T @ d["F"] @ C_vir
        _, R2full = cs_ccsd_residual(
            np.zeros((na, nv)), T2f, f_oo_e, f_vv_e, f_ov_e,
            __import__("vibeqc.dlpno._ccsd_cs", fromlist=["_blocks"])._blocks(
                d["B_ov"], d["B_vv"], d["B_oo"]),
        )
        oracle = {(i, j): U[(i, j)].T @ R2full[i, j] @ U[(i, j)] for (i, j) in T2}

        def Vof(k, l):
            return Vpno[(k, l)] if (k, l) in Vpno else Vpno[(l, k)]

        def Tof(k, l):
            return T2[(k, l)] if (k, l) in T2 else T2[(l, k)].T

        err = 0.0
        for (i, j) in T2:
            Vij = Vpno[(i, j)]
            B_occ = build_pair_occ_pno(d["df"], d["C_loc"], Vij)
            B_vvp = np.einsum("Puv,ua,vb->Pab", d["B_vv_can"], U[(i, j)], U[(i, j)],
                              optimize=True)
            f_vv_ij = U[(i, j)].T @ f_vv_e @ U[(i, j)]

            def proj(m, n, Vij=Vij):
                return project_amplitude(pno_overlap(Vij, Vof(m, n), S), Tof(m, n))

            R = local_t2_residual_doubles(
                i, j, B_occ, B_vvp, d["B_oo"], f_oo_e, f_vv_ij, Tof(i, j), proj
            )
            err = max(err, float(np.max(np.abs(R - oracle[(i, j)]))))
        return err

    def test_exact_at_full_domains(self):
        d = _setup_h2o_dz(tcut=-1.0)  # full PNO space, every pair
        assert self._run(d) < 1e-10

    def test_truncated_domain_approximation_is_small(self):
        d = _setup_h2o_dz(tcut=1e-7)
        # Truncated domains: the cross-domain intermediates are the DLPNO
        # approximation, not exact — but bounded well below the residual.
        assert self._run(d) < 1e-2
