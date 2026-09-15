"""M1 diagnostic — lattice 3c ERI tensor + MgO remeasurement.

2026-05-26: From handovers/HANDOVER_GDF_V0_11_2026_05_29.md, open items for M1:

1. Compare vibe-qc lattice-summed 3c (molecular → periodic) diff on LiH primitive.
   If molecular 3c matches PySCF bit-exact (verified), then any lattice-error
   would show in the lattice vs molecular diff.

2. Audit the chg-axis convention: check per-L scale factors on T_fused.

3. Re-measure MgO 8-atom conventional cell after the Bloch pair-FT fix.

This script does items 1 and 3 purely with vibe-qc (no PySCF subprocess needed).
"""

from __future__ import annotations

import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import (
    compute_3c_eri,
    compute_3c_eri_lattice,
    direct_lattice_cells,
)
from vibeqc.aux_basis import (
    fuse_transform_matrix,
    make_aux_basis_set,
    make_compensating_basis,
    make_fused_basis,
    make_modrho_aux_basis,
)

ANG2BOHR = 1.0 / 0.529177210903


# ─── LiH primitive FCC setup ────────────────────────────────────────
def _lih_setup(basis_name="sto-3g", aux_name="def2-svp-jk", eta=0.25):
    A_LIH = 4.084 * ANG2BOHR
    lattice = (
        np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        )
        * A_LIH
    )
    atoms = [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0.5 * A_LIH] * 3)]
    system = vq.PeriodicSystem(3, lattice, atoms)
    mol = system.unit_cell_molecule()
    ao = vq.BasisSet(mol, basis_name)
    aux = make_aux_basis_set(mol, aux_name=aux_name)
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=eta)
    fused = make_fused_basis(modrho, chg, mol)
    return system, mol, ao, aux, modrho, chg, fused


def _mgo_8atom_setup(basis_name="sto-3g", aux_name="def2-svp-jk"):
    """MgO 8-atom conv cell (rocksalt): 4 Mg + 4 O."""
    A_MGO = 4.212 * ANG2BOHR  # conventional-cell lattice constant
    lattice = np.eye(3) * A_MGO
    atoms = [
        vq.Atom(12, [0, 0, 0]),
        vq.Atom(12, [0.5 * A_MGO, 0.5 * A_MGO, 0]),
        vq.Atom(12, [0.5 * A_MGO, 0, 0.5 * A_MGO]),
        vq.Atom(12, [0, 0.5 * A_MGO, 0.5 * A_MGO]),
        vq.Atom(8, [0.5 * A_MGO, 0, 0]),
        vq.Atom(8, [0, 0.5 * A_MGO, 0]),
        vq.Atom(8, [0, 0, 0.5 * A_MGO]),
        vq.Atom(8, [0.5 * A_MGO, 0.5 * A_MGO, 0.5 * A_MGO]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    mol = system.unit_cell_molecule()
    ao = vq.BasisSet(mol, basis_name)
    aux = make_aux_basis_set(mol, aux_name=aux_name)
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=0.25)
    fused = make_fused_basis(modrho, chg, mol)
    return system, mol, ao, aux, modrho, chg, fused


def _shell_labels(basis):
    """Return list of (region_str, L) tags for each AO in basis."""
    labels = []
    for sh in basis.shells():
        L = int(sh.l)
        nc = 2 * L + 1
        labels.extend([f"L={L}"] * nc)
    return labels


def _per_L_frobenius(M, basis, axis=0):
    """Return dict {L: Frobenius norm} for each L along one axis."""
    n = M.shape[axis]
    labels = _shell_labels(basis)
    assert len(labels) == n, f"{len(labels)} != {n}"
    result = {}
    for L in sorted(set(lb.split("=")[-1] for lb in labels)):
        L_val = int(L.split("=")[-1])
        idxs = [i for i, lb in enumerate(labels) if lb.endswith(f"={L}")]
        if axis == 0:
            block = M[idxs]
        elif axis == 1:
            block = M[:, idxs]
        elif axis == 2:
            block = M[:, :, idxs]
        result[L_val] = float(np.linalg.norm(block))
    return result


# ─── Item 1: molecular vs lattice 3c on LiH ─────────────────────────
def item1_lattice_vs_molecular():
    print("=" * 72)
    print("Item 1: LiH primitive — molecular vs lattice 3c on fused basis")
    print("=" * 72)
    system, mol, ao, aux, modrho, chg, fused = _lih_setup()
    A = fuse_transform_matrix(modrho, chg)
    n_aux = modrho.nbasis
    n_chg = chg.nbasis
    n_fused = fused.nbasis
    n_orb = ao.nbasis

    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = 30.0
    cells = direct_lattice_cells(system, lo.cutoff_bohr)
    print(f"  n_cells = {len(cells)} at cutoff {lo.cutoff_bohr} bohr")
    print(f"  n_orb = {n_orb}, n_aux = {n_aux}, n_chg = {n_chg}, n_fused = {n_fused}")

    # vibe-qc molecular 3c (no lattice sum) on the fused basis
    T_mol = np.asarray(compute_3c_eri(ao, fused))
    T_mol = T_mol.reshape(n_orb, n_orb, n_fused).transpose(2, 0, 1)
    T_mol = 0.5 * (T_mol + np.swapaxes(T_mol, 1, 2))
    print(f"\n  molecular 3c shape: {T_mol.shape}")
    print(f"  ‖T_mol(aux)‖_F  = {np.linalg.norm(T_mol[:n_aux]):.6e}")
    print(f"  ‖T_mol(chg)‖_F  = {np.linalg.norm(T_mol[n_aux:]):.6e}")

    # vibe-qc lattice 3c
    T_lat = np.asarray(compute_3c_eri_lattice(ao, fused, system, lo))
    print(f"\n  lattice 3c shape: {T_lat.shape}")
    print(f"  ‖T_lat(aux)‖_F  = {np.linalg.norm(T_lat[:n_aux]):.6e}")
    print(f"  ‖T_lat(chg)‖_F  = {np.linalg.norm(T_lat[n_aux:]):.6e}")

    # Diff
    T_diff = T_lat - T_mol
    print(f"\n  lattice − molecular:")
    print(f"  ‖diff(aux)‖_F = {np.linalg.norm(T_diff[:n_aux]):.6e}")
    print(f"  ‖diff(chg)‖_F = {np.linalg.norm(T_diff[n_aux:]):.6e}")
    print(
        f"  ‖diff(T_total)‖_F / ‖T_lat‖_F = "
        f"{np.linalg.norm(T_diff) / np.linalg.norm(T_lat):.4e}"
    )

    # Per-orbital-L breakdown of diff on chg rows
    orb_labels = _shell_labels(ao)
    print(f"\n  AO labels: {orb_labels}")
    for L in sorted(set(lb.split("=")[-1] for lb in orb_labels)):
        idxs = [i for i, lb in enumerate(orb_labels) if lb.endswith(f"={L}")]
        print(f"    L={L} AOs: indices {idxs}")

    # Per-chg-shell Frobenius norm
    chg_labels = _shell_labels(chg)
    print(f"\n  chg shell labels: {chg_labels}")
    print(f"  Per-chg-AO lattice 3c Frobenius (each chg AO across all orb pairs):")
    for i, lb in enumerate(chg_labels):
        f_norm = float(np.linalg.norm(T_lat[n_aux + i]))
        f_mol = float(np.linalg.norm(T_mol[n_aux + i]))
        ratio = f_norm / f_mol if f_mol > 1e-30 else float("nan")
        print(
            f"    chg AO {i:>2} ({lb}): lat={f_norm:.4e}, mol={f_mol:.4e}, "
            f"lat/mol={ratio:.4f}"
        )

    # Contract to modrho basis
    T_aux = A @ T_lat.reshape(n_fused, n_orb * n_orb).reshape(n_fused, n_orb, n_orb)
    T_aux_mol = A @ T_mol.reshape(n_fused, n_orb * n_orb).reshape(n_fused, n_orb, n_orb)
    print(f"\n  After A-transform (fused→modrho):")
    print(f"  ‖T_aux(lat)‖_F = {np.linalg.norm(T_aux):.6e}")
    print(f"  ‖T_aux(mol)‖_F = {np.linalg.norm(T_aux_mol):.6e}")
    print(
        f"  ‖T_aux(lat) − T_aux(mol)‖_F / ‖T_aux(lat)‖_F = "
        f"{np.linalg.norm(T_aux - T_aux_mol) / np.linalg.norm(T_aux):.4e}"
    )

    return system, mol, ao, aux, modrho, chg, fused


# ─── Item 4: MgO 8-atom Γ-only re-measurement ───────────────────────
def item4_mgo_remeasure():
    print("\n" + "=" * 72)
    print("Item 4: MgO 8-atom Γ-only compcell SCF with AFT (post-Bloch-fix)")
    print("=" * 72)
    system, mol, ao, aux, modrho, chg, fused = _mgo_8atom_setup()
    print(f"  MgO 8-atom: n_orb={ao.nbasis}, n_aux={modrho.nbasis}, n_chg={chg.nbasis}")

    for apply_aft in [False, True]:
        for ft_conv in ["libcint", "libint"]:
            print(f"\n  apply_aft={apply_aft}, ft_convention={ft_conv}:")
            try:
                opts = vq.PeriodicRHFOptions()
                opts.use_diis = True
                opts.max_iter = 50
                opts.conv_tol_energy = 1e-7
                opts.conv_tol_grad = 1e-6
                opts.lattice_opts.cutoff_bohr = 30.0
                opts.lattice_opts.nuclear_cutoff_bohr = 30.0

                import time

                t0 = time.perf_counter()
                r = vq.run_pbc_gdf_rhf(
                    system,
                    ao,
                    options=opts,
                    aux_basis="def2-svp-jk",
                    compcell_eta=0.25,
                    apply_aft_correction=apply_aft,
                    aft_ft_convention=ft_conv,
                    progress=False,
                )
                dt = time.perf_counter() - t0
                print(f"    E_total = {r.energy:+.6f} Ha")
                print(f"    converged={r.converged}, n_iter={r.n_iter}, wall={dt:.1f}s")
                print(f"    backend={r.backend}")
            except Exception as exc:
                print(f"    FAILED: {type(exc).__name__}: {str(exc)[:200]}")


# ─── Item 2: chg-axis convention audit ──────────────────────────────
def item2_chg_convention_audit(system, mol, ao, modrho, chg, fused):
    print("\n" + "=" * 72)
    print("Item 2: chg-axis convention audit on T_fused (LiH primitive)")
    print("=" * 72)
    from vibeqc._aopair_ft import per_ao_libint_norm_factor

    n_aux = modrho.nbasis
    n_chg = chg.nbasis
    n_fused = fused.nbasis
    n_orb = ao.nbasis

    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = 30.0
    T_fused = np.asarray(compute_3c_eri_lattice(ao, fused, system, lo))
    T_chg = T_fused[n_aux:]  # (n_chg, n_orb, n_orb)

    # Per-L Frobenius of T_chg for each L on the chg axis
    print("\n  Per-chg-L Frobenius of T_chg[L, :, :]:")
    chg_labels = _shell_labels(chg)
    for L in sorted(set(lb.split("=")[-1] for lb in chg_labels)):
        L_val = int(L.split("=")[-1])
        idxs = [i for i, lb in enumerate(chg_labels) if lb.endswith(f"={L}")]
        f_norm = float(np.linalg.norm(T_chg[idxs]))
        print(f"    chg L={L_val} ({len(idxs)} AOs): ‖T_chg‖_F = {f_norm:.6e}")

    # libint per-AO norm factors
    print("\n  libint per-AO norm factors (for the fused basis):")
    for i, sh in enumerate(fused.shells()):
        L = int(sh.l)
        scale = np.sqrt(4.0 * np.pi / (2 * L + 1))
        bf_start = sum(2 * int(s.l) + 1 for s in list(fused.shells())[:i])
        region = "aux" if bf_start < n_aux else "chg"
        print(
            f"    shell {i}: {region}, L={L}, "
            f"scale=√(4π/(2L+1)) = {scale:.6f}, "
            f"n_AOs={2 * L + 1}"
        )

    # Per-AO variance of T_fused along the pair axes
    # For chg AO 0 (Li L=0 at origin), T_fused should be dominated by
    # the home-cell contribution. Print a slice.
    print("\n  T_fused[chg AO 0] slice (Li L=0, at origin): μ=0..3, ν=0..3:")
    chg0 = T_fused[n_aux]  # first chg AO
    print(f"    {chg0[:4, :4]}")
    print(f"  T_fused[chg AO 16] slice (H L=0, H is at 1/2,1/2,1/2): μ=0..3, ν=0..3:")
    chg16 = T_fused[n_aux + 16]
    print(f"    {chg16[:4, :4]}")


# ─── Main ───────────────────────────────────────────────────────────
def main():
    print(f"vibeqc {vq.__version__}: M1 GDF diagnostic — lattice 3c + MgO")
    print()

    # Item 1: molecular vs lattice 3c diff
    system, mol, ao, aux, modrho, chg, fused = item1_lattice_vs_molecular()

    # Item 2: chg-axis convention audit
    item2_chg_convention_audit(system, mol, ao, modrho, chg, fused)

    # Item 4: MgO remeasurement
    item4_mgo_remeasure()


if __name__ == "__main__":
    main()
