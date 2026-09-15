"""Profile vibe-qc PBCGDF compcell SCF to identify the hot path.

Measurements per stage on H2 / def2-svp-jk and on a larger 8-atom
LiH-like vacuum cluster. Reports wall time per call and per-iter
breakdown to guide optimization.

The expensive parts of compcell GDF:
  - Build Lpq once: 2c lattice sum on fused basis (libint), 3c
    lattice sum on (orbital, fused) (libint), eigendecompose +
    threshold, einsum to form Lpq.
  - Per SCF iter: J einsum O(N_aux · N_orb²), K einsum
    O(N_aux · N_orb² · N_orb), diagonalize O(N_orb³), DIIS extrapolate
    O(N_orb²).
"""
from __future__ import annotations

import time

import numpy as np

import vibeqc as vq
from vibeqc.aux_basis import (
    build_lpq_compcell,
    make_aux_basis_set,
    make_modrho_aux_basis,
    make_compensating_basis,
    make_fused_basis,
    fuse_transform_matrix,
)
from vibeqc._vibeqc_core import (
    compute_2c_eri_lattice,
    compute_3c_eri_lattice,
)


def _h2_box(box_bohr: float = 12.0, sep_bohr: float = 1.4):
    half = 0.5 * sep_bohr
    system = vq.PeriodicSystem(
        3, np.diag([box_bohr, box_bohr, box_bohr]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _h8_vacuum_box(box_bohr: float = 16.0, sep_bohr: float = 1.4):
    """8 H atoms in 4 H2 pairs spread across a vacuum box. Larger than
    H2 for SCF cost scaling; still molecular (vacuum-padded) so we
    don't fight the LiH-style ionic compcell divergence."""
    half = 0.5 * sep_bohr
    L = box_bohr
    # 4 pairs at the corners of a tetrahedron offset
    centers = [(0.3*L, 0.3*L, 0.3*L), (0.7*L, 0.7*L, 0.3*L),
               (0.3*L, 0.7*L, 0.7*L), (0.7*L, 0.3*L, 0.7*L)]
    atoms = []
    for cx, cy, cz in centers:
        atoms.append(vq.Atom(1, [cx, cy, cz - half]))
        atoms.append(vq.Atom(1, [cx, cy, cz + half]))
    system = vq.PeriodicSystem(3, np.diag([L, L, L]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


class Timer:
    def __init__(self, label):
        self.label = label
        self.t0 = None
        self.dt = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.dt = time.perf_counter() - self.t0
        print(f"  {self.label:<40s}  {self.dt*1000:>10.2f} ms")


def profile_build_lpq(system, basis):
    """Break down build_lpq_compcell into its expensive stages."""
    mol = system.unit_cell_molecule()
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = 30.0
    lo.nuclear_cutoff_bohr = 30.0

    with Timer("make_modrho_aux_basis"):
        aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    with Timer("make_compensating_basis"):
        modrho = make_modrho_aux_basis(aux, mol)
    with Timer("make_compensating_basis(eta=0.25)"):
        chg = make_compensating_basis(modrho, mol, eta=0.25)
    with Timer("make_fused_basis"):
        fused = make_fused_basis(modrho, chg, mol)
    with Timer("fuse_transform_matrix"):
        A = fuse_transform_matrix(modrho, chg)
    with Timer("compute_2c_eri_lattice(fused)"):
        M_fused = np.asarray(compute_2c_eri_lattice(fused, system, lo))
    with Timer("compute_3c_eri_lattice(orbital, fused)"):
        T_fused = np.asarray(compute_3c_eri_lattice(basis, fused, system, lo))
    with Timer("A @ M_fused @ A.T"):
        M_compcell = A @ M_fused @ A.T
        M_compcell = 0.5 * (M_compcell + M_compcell.T)
    with Timer("einsum('iP,Pmn->imn', A, T_fused)"):
        T_compcell = np.einsum("iP,Pmn->imn", A, T_fused, optimize=True)
    with Timer("eigh(M_compcell)"):
        eigvals, U = np.linalg.eigh(M_compcell)
    with Timer("U.T @ T_flat / sqrt(eig)"):
        n_aux = modrho.nbasis
        n_orb = basis.nbasis
        keep = eigvals > 1e-9 * eigvals[-1]
        T_flat = T_compcell.reshape(n_aux, n_orb * n_orb)
        Lpq_flat = (U[:, keep].T @ T_flat) / np.sqrt(eigvals[keep])[:, None]
        Lpq = Lpq_flat.reshape(int(keep.sum()), n_orb, n_orb)
    return Lpq


def profile_per_iter(Lpq, basis):
    """Per-iter J/K build cost from a precomputed Lpq."""
    n_orb = basis.nbasis
    # Make a random-ish closed-shell D for benchmark
    rng = np.random.default_rng(0)
    C_occ = rng.standard_normal((n_orb, n_orb // 2))
    D = 2.0 * (C_occ @ C_occ.T)
    D = 0.5 * (D + D.T)

    # Warm-up cache + run 5 iters
    times_J = []
    times_K = []
    for trial in range(5):
        with Timer(f"  iter J  (trial {trial+1})") as t_j:
            rho = np.einsum("Lij,ij->L", Lpq, D, optimize=True)
            J = np.einsum("L,Lij->ij", rho, Lpq, optimize=True)
        times_J.append(t_j.dt)
        with Timer(f"  iter K  (trial {trial+1})") as t_k:
            K = np.einsum("Lmk,kl,Lnl->mn", Lpq, D, Lpq, optimize=True)
        times_K.append(t_k.dt)
    print(f"  median per-iter J: {np.median(times_J)*1000:>10.2f} ms")
    print(f"  median per-iter K: {np.median(times_K)*1000:>10.2f} ms")
    print(f"  Lpq.shape = {Lpq.shape}, n_orb²·n_aux = {n_orb**2 * Lpq.shape[0]}")


def main() -> int:
    print(f"vibeqc {vq.__version__}: PBCGDF compcell hot-path profile")
    print()

    for label, builder in [("H2 / 12-bohr / sto-3g", _h2_box),
                           ("H8 / 16-bohr / sto-3g", _h8_vacuum_box)]:
        print(f"=== {label} ===")
        system, basis = builder()
        print(f"  n_orb = {basis.nbasis}, n_atoms = {sum(1 for _ in system.unit_cell)}")
        print(f"  --- Lpq build stages ---")
        Lpq = profile_build_lpq(system, basis)
        print(f"  --- Per-iter J/K (5 trials, median) ---")
        profile_per_iter(Lpq, basis)
        print()

    # AFT-enabled path: measure the additional cost of
    # _compcell_aft_correction(_3c)
    print("=== AFT-enabled path (H2 / def2-svp-jk, eta=1.0) ===")
    from vibeqc.aux_basis import (
        _compcell_aft_correction, _compcell_aft_correction_3c,
    )
    system, basis = _h2_box()
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=1.0)
    fused = make_fused_basis(modrho, chg, mol)
    n_aux = modrho.nbasis
    with Timer("_compcell_aft_correction (2c)"):
        j2c_p = _compcell_aft_correction(fused, n_aux, system, eta=1.0,
                                          ft_convention="libint")
    with Timer("_compcell_aft_correction_3c"):
        j3c_p = _compcell_aft_correction_3c(fused, basis, n_aux, system,
                                             eta=1.0, ft_convention="libint")
    print(f"  j2c_p shape: {j2c_p.shape}; j3c_p shape: {j3c_p.shape}")
    print()

    # Full-pipeline timing for comparison
    print("=== Full run_pbc_gdf_rhf timing (H2 / 12-bohr) ===")
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True; opts.damping = 0.0; opts.max_iter = 40
    opts.conv_tol_energy = 1e-10
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    for label, kw in [("no AFT", {}),
                      ("AFT(libint) + auto-rcut",
                       {"apply_aft_correction": True,
                        "aft_ft_convention": "libint",
                        "rcut_strategy": "pyscf_auto"})]:
        t0 = time.perf_counter()
        r = vq.run_pbc_gdf_rhf(system, basis, opts, aux_basis="def2-svp-jk",
                               exxdiv="ewald", compcell_eta=1.0,
                               progress=False, **kw)
        dt = time.perf_counter() - t0
        print(f"  {label:<40s}  {dt*1000:>10.2f} ms  "
              f"({r.n_iter} iter, conv={r.converged})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
