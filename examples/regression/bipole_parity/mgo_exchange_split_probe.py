"""MgO Γ Ewald-exchange-split reconstruction probe vs PySCF (out-of-process).

The Phase-1 tool of the option (b) energy-assembly redesign
(2026-06-10): validates, element-wise against PySCF GDF, the BIPOLE
exchange convention

    K = K_SR(erfc ω, direct) + K_LR(erf ω, reciprocal, K≠0)
        + c·S·D·S,   c = −π/(Vω²) [exxdiv=None]
                     c = ξ_M − π/(Vω²) [exxdiv='ewald']

and the J-side identity (the v_bg·S background term IS the G=0 removal
of J_SR):

    J_SR(erfc) + J_LR(erf, K≠0) − (π·N_e/(Vω²))·S ≈ vj(PySCF)

Findings on MgO/STO-3G Γ at the PySCF converged density (2026-06-10,
PySCF 2.13, builder cutoff 14 / density list 30 bohr):

  * |ΔK|_max = 8.2e-4, ΔE_K = −0.32 mHa — both exxdiv conventions
    (identical residual, as they must: they differ by the exact ξ·SDS).
  * |ΔJ|_max = 1.5e-3, ΔE_J = +0.41 mHa.
  * ω-invariant: ω → 1.3ω moves E_K(reconstructed) by 0.4 mHa while
    the SR/LR pieces redistribute by ~1.3 Ha.
  * PySCF identity vk(ewald) − vk(None) = ξ_M·S·D·S to 9e-16.

The direct-space full-Coulomb K with the Γ Bloch density (the pre-fix
convention) is formally divergent — its value at any cutoff is a
truncation artefact (−75.7 at cutoff 10 on this system).

Stage 1 (PySCF, own interpreter — never imported by vibe-qc,
CLAUDE.md §10). Requires /tmp/mgo_pyscf_gamma.npz from
``mgo_component_audit.py pyscf`` (for the converged density):

    python mgo_exchange_split_probe.py pyscf /tmp/mgo_pyscf_jk2.npz

Stage 2 (vibe-qc):

    python mgo_exchange_split_probe.py vibeqc /tmp/mgo_pyscf_jk2.npz \
        [cut_builder=14] [cut_density=30] [omega_scale=1.0]
"""

from __future__ import annotations

import sys

import numpy as np

A_ANG = 4.21
ANG2BOHR = 1.0 / 0.529177210903


def _pyscf_cell():
    from pyscf.pbc import gto as pbc_gto

    half = A_ANG / 2.0
    lattice_ang = (A_ANG / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    return pbc_gto.M(
        atom=f"Mg 0 0 0; O {half} {half} {half}",
        a=lattice_ang.tolist(),
        basis="sto-3g",
        unit="A",
        verbose=0,
    )


def stage_pyscf(out_path: str) -> None:
    from pyscf.pbc import scf as pbc_scf
    from pyscf.pbc import tools as pbc_tools

    cell = _pyscf_cell()
    dat = np.load("/tmp/mgo_pyscf_gamma.npz")
    D = dat["D"]

    mf = pbc_scf.RHF(cell).density_fit()
    mf.exxdiv = None
    vj, vk_none = mf.get_jk(dm=D)
    mf.exxdiv = "ewald"
    _, vk_ewald = mf.get_jk(dm=D)
    xi = pbc_tools.madelung(cell, np.zeros((1, 3)))
    S = mf.get_ovlp()

    sds = S @ D @ S
    resid = np.abs(vk_ewald - vk_none - xi * sds).max()
    print(f"E_K(None)  = {-0.25 * float(np.einsum('ij,ji->', D, vk_none).real):.12f}")
    print(f"E_K(ewald) = {-0.25 * float(np.einsum('ij,ji->', D, vk_ewald).real):.12f}")
    print(f"madelung   = {xi:.12f}   |vk_ew - vk_none - xi*SDS| = {resid:.2e}")
    np.savez(
        out_path,
        D=D, S=S, vj=vj, vk_none=vk_none, vk_ewald=vk_ewald, madelung=xi,
    )
    print(f"-> {out_path}")


def stage_vibeqc(npz_path: str, cut_builder: float, cut_density: float,
                 omega_scale: float) -> None:
    import vibeqc as vq
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        build_jk_2e_real_space,
        compute_overlap_lattice,
    )
    from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
    from vibeqc.bipole_fock_ewald import (
        _build_j_long_range_cache,
        compute_K_long_range_gamma,
    )

    a = A_ANG * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    sysp = vq.PeriodicSystem(
        3, lattice,
        [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis
    V = float(abs(np.linalg.det(lattice)))

    dat = np.load(npz_path)
    # AO map: PySCF groups s before p per atom; libint uses declaration
    # order (validated via S(Γ), 2026-06-10).
    spec = [0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10, 11, 12, 13]
    P = np.zeros((nbf, nbf))
    for v_i, p_i in enumerate(spec):
        P[v_i, p_i] = 1.0
    D = P @ dat["D"] @ P.T
    vj = P @ dat["vj"] @ P.T
    vk_none = P @ dat["vk_none"] @ P.T
    vk_ewald = P @ dat["vk_ewald"] @ P.T
    S_py = P @ dat["S"] @ P.T
    xi = float(dat["madelung"])

    omega = crystal_default_ewald_alpha(V) * omega_scale
    print(f"V={V:.2f}  omega={omega:.4f}  cut_builder={cut_builder}  "
          f"cut_density={cut_density}")

    # Bloch density: D at EVERY cell of a wide list (P(h) must be
    # resolvable for every difference the builder's traversal forms).
    lat_d = LatticeSumOptions()
    lat_d.cutoff_bohr = cut_density
    lat_d.nuclear_cutoff_bohr = cut_density
    dens = compute_overlap_lattice(basis, sysp, lat_d)
    for c in range(len(dens.cells)):
        dens.set_block(c, D)

    lat_b = LatticeSumOptions()
    lat_b.cutoff_bohr = cut_builder
    lat_b.nuclear_cutoff_bohr = cut_builder
    S_lat = compute_overlap_lattice(basis, sysp, lat_b)
    S_gamma = np.zeros((nbf, nbf))
    for c in range(len(S_lat.cells)):
        S_gamma += np.asarray(S_lat.blocks[c], dtype=float)

    jk = build_jk_2e_real_space(basis, sysp, lat_b, dens, omega)
    K_SR = np.zeros((nbf, nbf))
    J_SR = np.zeros((nbf, nbf))
    for c in range(len(jk.K.cells)):
        K_SR += np.asarray(jk.K.blocks[c], dtype=float)
        J_SR += np.asarray(jk.J.blocks[c], dtype=float)

    cells_r_cart = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in S_lat.cells], dtype=float
    )
    cache = _build_j_long_range_cache(basis, sysp, cells_r_cart, omega, 1e-8)
    ft_bloch = cache.ft_per_cell.sum(axis=0)
    K_LR = compute_K_long_range_gamma(cache, D)

    rho_hat = np.einsum("mn,mnk->k", D, ft_bloch, optimize=True)
    J_LR = np.einsum(
        "k,mnk->mn", cache.kernel * rho_hat, ft_bloch.conj(), optimize=True
    ).real
    n_elec = float(np.einsum("ij,ji->", D, S_gamma))
    J_recon = J_SR + J_LR - (np.pi * n_elec / (V * omega * omega)) * S_gamma
    ej = 0.5 * float(np.einsum("ij,ji->", D, J_recon))
    ej_ref = 0.5 * float(np.einsum("ij,ji->", D, vj))
    print(f"J : E_J={ej:+.9f} ref={ej_ref:+.9f} dE={ej - ej_ref:+.2e} "
          f"|dJ|={np.abs(J_recon - vj).max():.2e}")

    g0 = np.pi / (V * omega * omega)
    SDS = S_py @ D @ S_py
    for name, mine, ref in (
        ("None ", K_SR + K_LR - g0 * SDS, vk_none),
        ("ewald", K_SR + K_LR + (xi - g0) * SDS, vk_ewald),
    ):
        ek = -0.25 * float(np.einsum("ij,ji->", D, mine))
        ek_ref = -0.25 * float(np.einsum("ij,ji->", D, ref))
        print(f"K {name}: E_K={ek:+.9f} ref={ek_ref:+.9f} "
              f"dE={ek - ek_ref:+.2e} |dK|={np.abs(mine - ref).max():.2e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "vibeqc"
    path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mgo_pyscf_jk2.npz"
    if mode == "pyscf":
        stage_pyscf(path)
    else:
        cb = float(sys.argv[3]) if len(sys.argv) > 3 else 14.0
        cd = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
        ws = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        stage_vibeqc(path, cb, cd, ws)
