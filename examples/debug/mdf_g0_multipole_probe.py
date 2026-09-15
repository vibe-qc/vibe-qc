"""Diagnostic harness for the MDF Eq-23 G=0 self-term higher multipoles.

Measures the EXACT per-aux gap

    gap[P, mu, nu] = T_real[P, mu, nu] - T_recip_dense[P, mu, nu]

where ``T_real`` is the real-space libint 3c on the compensated fused
basis (A-transformed) and ``T_recip_dense`` is the reciprocal-space PW
projection  Sum_{G != 0} coul(G) rho_P(-G) rho_munu(G)  on a DENSE mesh.
This gap IS the Ewald-regularised G=0 self-term V-bar.rho-bar that the
MDF builder must add back (docs/design_mdf.md sec 4).

The current production code reproduces only the MONOPOLE piece
    V-bar_P = -pi^{5/2} (A @ m_fused),  rho-bar_munu = S_munu / Omega
(s-type fused fns x AO-pair overlap). This probe isolates the residual
    residual = gap - monopole
and decomposes it against candidate higher-multipole structures, so the
dipole/quadrupole coefficients are PINNED to the measured gap, not
guessed (CLAUDE.md sec 7).

Run:  .venv/bin/python examples/debug/mdf_g0_multipole_probe.py
"""

from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc.aux_basis import (
    fuse_transform_matrix,
    make_compensating_basis,
    make_fused_basis,
    make_modrho_aux_basis,
    make_aux_basis_set,
    rsgdf_aux_fourier_transform,
    rsgdf_dense_g_mesh,
    _ao_scales_for_rsgdf,
)
from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch
from vibeqc._vibeqc_core import (
    bloch_sum,
    compute_3c_eri_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
)


def _setup(system, ao_basis, aux_basis, eta, cutoff_bohr):
    mol = system.unit_cell_molecule()
    lat = vq.LatticeSumOptions()
    lat.cutoff_bohr = cutoff_bohr
    lat.nuclear_cutoff_bohr = cutoff_bohr

    modrho_aux = make_modrho_aux_basis(aux_basis, mol)
    chg = make_compensating_basis(modrho_aux, mol, eta=eta)
    fused = make_fused_basis(modrho_aux, chg, mol)
    A = fuse_transform_matrix(modrho_aux, chg)
    return mol, lat, modrho_aux, chg, fused, A


def compute_gap(system, ao_basis, aux_basis, *, eta=1.0, cutoff_bohr=30.0, ke=400.0):
    """Return (gap, T_real, pw_dense, A, fused, modrho_aux, S_ao, V, lat)."""
    mol, lat, modrho_aux, chg, fused, A = _setup(
        system, ao_basis, aux_basis, eta, cutoff_bohr
    )
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))

    # Real-space libint 3c on compensated fused basis.
    T_fused = np.asarray(compute_3c_eri_lattice(ao_basis, fused, system, lat))
    T = np.einsum("iP,Pmn->imn", A, T_fused, optimize=True)

    # Dense reciprocal PW projection (G != 0).
    G_all = rsgdf_dense_g_mesh(system, ke)
    G2 = (G_all**2).sum(axis=1)
    nz = G2 > 1e-12
    G = G_all[nz]
    coul = (4.0 * np.pi) / G2[nz] / V

    aux_ft_fused = rsgdf_aux_fourier_transform(fused, G)
    rho_aux = A @ aux_ft_fused  # (n_aux, n_G)

    cells = direct_lattice_cells(system, float(lat.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)
    pair_ft = ao_pair_fourier_transform_bloch(ao_basis, G, R_g, k_cart=np.zeros(3))
    ao_scales = _ao_scales_for_rsgdf(ao_basis)
    pair_ft = pair_ft * np.outer(ao_scales, ao_scales)[:, :, None]

    aux_w = rho_aux.conj() * coul[None, :]
    pw_dense = np.real(np.einsum("Pk,mnk->Pmn", aux_w, pair_ft, optimize=True))
    pw_dense = 0.5 * (pw_dense + pw_dense.transpose(0, 2, 1))

    gap = T - pw_dense

    S_ao = np.real(bloch_sum(compute_overlap_lattice(ao_basis, system, lat), np.zeros(3)))
    S_ao = 0.5 * (S_ao + S_ao.T)
    return dict(
        gap=gap, T=T, pw_dense=pw_dense, A=A, fused=fused,
        modrho_aux=modrho_aux, S_ao=S_ao, V=V, lat=lat, mol=mol,
    )


def monopole_term(fused, A, S_ao, V):
    m_fused = np.zeros(fused.nbasis)
    off = 0
    for sh in fused.shells():
        nb = (2 * sh.l + 1) if sh.pure else ((sh.l + 1) * (sh.l + 2) // 2)
        if sh.l == 0:
            a = np.asarray(sh.exponents, dtype=float)
            c = np.asarray(sh.coefficients, dtype=float)
            m_fused[off : off + nb] = float(np.sum(c * a ** (-2.5)))
        off += nb
    vbar = -(np.pi**2.5) * (A @ m_fused)
    return vbar[:, None, None] * (S_ao / V)[None, :, :]


def aux_function_moments_via_ft(fused, A, lmax=2):
    """Extract single-aux-function Cartesian moments m0, d_a, Q_ab of the
    COMPENSATED aux phi_P = (A @ fused) from the analytic FT, by a
    high-order central finite-difference in G around 0.

    rho_P(G) = m0 - i (G.d) - 1/2 G_a G_b Q_ab + ...
      m0    =  Re rho(0)
      d_a   =  Im[ d rho / dG_a ]            (rho = m0 - i G.d + ... )
      Q_ab  = -Re[ d^2 rho / dG_a dG_b ]
    """
    h = 1e-3
    naux = A.shape[0]

    def rho(Gvec):
        ft = rsgdf_aux_fourier_transform(fused, np.atleast_2d(Gvec))
        return (A @ ft)[:, 0]

    e = np.eye(3)
    m0 = np.real(rho(np.zeros(3)))
    d = np.zeros((naux, 3))
    Q = np.zeros((naux, 3, 3))
    for a in range(3):
        rp = rho(h * e[a])
        rm = rho(-h * e[a])
        d[:, a] = np.imag((rp - rm) / (2 * h))
        Q[:, a, a] = -np.real((rp - 2 * m0 - rm + 2 * m0) / 1.0)  # placeholder
        Q[:, a, a] = -np.real((rp - 2 * rho(np.zeros(3)) + rm) / h**2)
    for a in range(3):
        for b in range(a + 1, 3):
            rpp = rho(h * (e[a] + e[b]))
            rpm = rho(h * (e[a] - e[b]))
            rmp = rho(h * (-e[a] + e[b]))
            rmm = rho(h * (-e[a] - e[b]))
            mixed = np.real((rpp - rpm - rmp + rmm) / (4 * h**2))
            Q[:, a, b] = -mixed
            Q[:, b, a] = -mixed
    return m0, d, Q


def ao_pair_moments(ao_basis, system, lat, lmax=2):
    """Periodic (Gamma Bloch-summed) AO-pair Cartesian moments around 0:
       S_munu, D_munu[a] = sum_R <mu| r_a |nu(R)>, Q_munu[a,b].
    Uses compute_multipole_moments_lattice (libint emultipole) summed
    over cells at k=0 (Gamma)."""
    from vibeqc._vibeqc_core import compute_multipole_moments_lattice
    from vibeqc.bipole_cell_moments import cartesian_component_indices

    mset = compute_multipole_moments_lattice(ao_basis, system, lat, lmax, [0.0, 0.0, 0.0])
    idxs = cartesian_component_indices(lmax)
    n = ao_basis.nbasis
    # Gamma Bloch sum = plain sum over cells (phase = 1).
    summed = []
    for comp in range(len(idxs)):
        acc = np.zeros((n, n), dtype=float)
        for c in range(len(mset.cells)):
            acc += np.asarray(mset.blocks[c][comp], dtype=float)
        summed.append(acc)
    summed = np.array(summed)  # (n_comp, n, n)

    def comp(i, j, k):
        return summed[idxs.index((i, j, k))]

    S = comp(0, 0, 0)
    D = np.stack([comp(1, 0, 0), comp(0, 1, 0), comp(0, 0, 1)], axis=0)  # (3,n,n)
    Q = np.zeros((3, 3, S.shape[0], S.shape[1]))
    Q[0, 0] = comp(2, 0, 0)
    Q[1, 1] = comp(0, 2, 0)
    Q[2, 2] = comp(0, 0, 2)
    Q[0, 1] = Q[1, 0] = comp(1, 1, 0)
    Q[0, 2] = Q[2, 0] = comp(1, 0, 1)
    Q[1, 2] = Q[2, 1] = comp(0, 1, 1)
    return S, D, Q


def probe_gap_residual(system, ao, aux, label):
    print("=" * 70)
    print(f"SYSTEM: {label}")
    print("=" * 70)

    # Mesh convergence of the gap.
    res200 = compute_gap(system, ao, aux, ke=200.0)
    res400 = compute_gap(system, ao, aux, ke=400.0)
    gap200, gap400 = res200["gap"], res400["gap"]
    print(f"n_aux = {res400['A'].shape[0]}, n_orb = {ao.nbasis}, V = {res400['V']:.3f}")
    print(f"|gap(ke=400)|max          = {np.abs(gap400).max():.6e}")
    print(f"|gap(400)-gap(200)|max    = {np.abs(gap400-gap200).max():.6e}  (mesh conv)")

    mono = monopole_term(res400["fused"], res400["A"], res400["S_ao"], res400["V"])
    residual = gap400 - mono
    print(f"|monopole|max             = {np.abs(mono).max():.6e}")
    print(f"|gap|max                  = {np.abs(gap400).max():.6e}")
    print(f"|residual=gap-mono|max    = {np.abs(residual).max():.6e}")
    print(f"residual / gap (rel)      = {np.abs(residual).max()/np.abs(gap400).max():.4%}")

    # Which aux P carry the residual? Print per-shell residual norm + aux L.
    print("\nPer-aux residual norm (and aux shell L):")
    off = 0
    modrho = res400["modrho_aux"]
    aux_L = []
    for sh in modrho.shells():
        nb = (2 * sh.l + 1) if sh.pure else ((sh.l + 1) * (sh.l + 2) // 2)
        for _ in range(nb):
            aux_L.append(int(sh.l))
        off += nb
    aux_L = np.array(aux_L)
    rper = np.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1)
    mper = np.linalg.norm(mono.reshape(mono.shape[0], -1), axis=1)
    for L in sorted(set(aux_L.tolist())):
        sel = aux_L == L
        print(f"  L={L}: count={sel.sum():3d}  "
              f"|residual|={np.linalg.norm(rper[sel]):.4e}  "
              f"|monopole|={np.linalg.norm(mper[sel]):.4e}")

    # Candidate higher-multipole structures.
    S, D, Q = ao_pair_moments(ao, system, res400["lat"], lmax=2)
    m0_P, d_P, Q_P = aux_function_moments_via_ft(res400["fused"], res400["A"])
    V = res400["V"]
    print("\nAux compensated-function moments (max abs over P):")
    print(f"  |m0_P|max = {np.abs(m0_P).max():.3e}  (monopole; ~0 expected)")
    print(f"  |d_P|max  = {np.abs(d_P).max():.3e}  (dipole)")
    print(f"  |Q_P|max  = {np.abs(Q_P).max():.3e}  (2nd moment)")

    # Dipole candidate:  (4pi/3V) d_P . D_munu
    dip = (4.0 * np.pi) / (3.0 * V) * np.einsum("Pa,amn->Pmn", d_P, D, optimize=True)
    print(f"\nDipole candidate (4pi/3V) d_P.D : |max|={np.abs(dip).max():.4e}")
    print(f"  |residual - dipole|max = {np.abs(residual - dip).max():.4e}")
    print(f"  reduces residual?       {np.abs(residual-dip).max() < np.abs(residual).max()}")

    # Least-squares scale for the dipole candidate (is the 1/3 right?).
    if np.abs(dip).max() > 1e-14:
        num = float(np.sum(residual * dip))
        den = float(np.sum(dip * dip))
        print(f"  best-fit scale on dipole = {num/den:.6f}  (1.0 = coefficient right)")
        res_after_dip = residual - (num / den) * dip
        print(f"  |residual after dip-fit|max = {np.abs(res_after_dip).max():.4e}")


def probe_W_vs_mesh(system, ao, aux, label):
    """Localize the W(MDF) vs W(rsgdf) plateau: does MDF converge to the
    dense rsgdf W as its OWN PW mesh tightens, or plateau (=> not a mesh
    or G=0 effect but the aux-representation/fit difference)?"""
    from vibeqc.aux_basis import build_lpq_mdf, build_lpq_native_fft

    mol = system.unit_cell_molecule()
    lat = vq.LatticeSumOptions()
    lat.cutoff_bohr = 30.0
    lat.nuclear_cutoff_bohr = 30.0
    n = ao.nbasis

    print("\n" + "-" * 70)
    print(f"W(MDF, ke) vs W(rsgdf dense ke=400) :  {label}")
    print("-" * 70)
    Lr = build_lpq_native_fft(
        system, ao, make_modrho_aux_basis(aux, mol), ke_cutoff=400.0, lat_opts=lat
    ).reshape(-1, n * n)
    W_ref = Lr.T @ Lr

    # rsgdf at the SAME modest ke (isolates aux-rep vs mesh).
    for ke in (40.0, 100.0, 200.0, 400.0):
        c = build_lpq_mdf(
            system, ao, aux, molecule=mol, lat_opts=lat,
            compcell_eta=1.0, mdf_ke_cutoff=ke,
        )
        Lg = c.L_gauss.reshape(c.n_kept_gauss, n * n)
        Wp = c.cderi_pw.reshape(c.n_pw, n * n)
        W_mdf = Lg.T @ Lg + np.real(Wp.conj().T @ Wp)
        Lr_ke = build_lpq_native_fft(
            system, ao, make_modrho_aux_basis(aux, mol), ke_cutoff=ke, lat_opts=lat
        ).reshape(-1, n * n)
        W_rs_ke = Lr_ke.T @ Lr_ke
        print(f"  ke={ke:6.0f}: |W(MDF)-W(rsgdf400)|max={np.abs(W_mdf-W_ref).max():.4e}"
              f"   |W(rsgdf ke)-W(rsgdf400)|max={np.abs(W_rs_ke-W_ref).max():.4e}")


def main():
    np.set_printoptions(precision=6, suppress=True, linewidth=140)

    # H2 box (matches tests).
    box = 12.0
    sep = 1.4
    h2 = vq.PeriodicSystem(
        3, np.diag([box, box, box]),
        [vq.Atom(1, [0, 0, -sep / 2]), vq.Atom(1, [0, 0, sep / 2])],
    )
    h2_ao = vq.BasisSet(h2.unit_cell_molecule(), "sto-3g")
    h2_aux = make_aux_basis_set(h2.unit_cell_molecule(), aux_name="def2-svp-jk")

    # Ne box (steep all-electron core — the actual MDF target).
    ne = vq.PeriodicSystem(3, np.diag([10.0] * 3), [vq.Atom(10, [0, 0, 0])])
    ne_ao = vq.BasisSet(ne.unit_cell_molecule(), "sto-3g")
    ne_aux = make_aux_basis_set(ne.unit_cell_molecule(), aux_name="def2-svp-jk")

    probe_gap_residual(h2, h2_ao, h2_aux, "H2 / sto-3g / def2-svp-jk, 12-bohr box")
    probe_W_vs_mesh(h2, h2_ao, h2_aux, "H2")
    print()
    probe_gap_residual(ne, ne_ao, ne_aux, "Ne / sto-3g / def2-svp-jk, 10-bohr box")
    probe_W_vs_mesh(ne, ne_ao, ne_aux, "Ne")


if __name__ == "__main__":
    main()
