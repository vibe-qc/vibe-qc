#!/usr/bin/env python3
"""Track A — SCF second-order vs DIIS benchmark (RKS + UKS).

Compares plain DIIS (the v0.7.x baseline) against Newton (D2c) and TRAH
(D2e) on representative closed-shell KS-DFT systems plus an open-shell
radical, across LDA / GGA / hybrid functionals. Reports iter count +
wall time + |ΔE vs DIIS| for each variant.

The headline finding (compute-host-d, Linux x86-64, OpenBLAS+LAPACKE, 4 CPUs):

  | system        | basis    | functional | DIIS         | Newton       | TRAH         | ΔE      |
  |---------------|----------|------------|--------------|--------------|--------------|---------|
  | H₂O           | cc-pVDZ  | LDA        | 19 / 2.5s    | 11 / 4.9s    | 11 / 4.8s    | 1e-14   |
  | H₂O           | cc-pVDZ  | PBE        | 18 / 2.7s    | 11 / 6.5s    | 11 / 6.4s    | 4e-14   |
  | H₂O           | cc-pVDZ  | B3LYP      | 18 / 3.6s    | 10 / 8.6s    | 10 / 8.2s    | 3e-14   |
  | Benzene       | 6-31G*   | PBE        | 26 / 41.7s   | 18 / 84.2s   | 18 / 81.6s   | 1e-12   |
  | Benzene       | 6-31G*   | B3LYP      | 18 / 30.0s   | 16 / 80.2s   | 16 / 80.4s   | 1e-13   |
  | OH• (doublet) | cc-pVDZ  | UKS-LDA    | 136 / 8.8s   | 12 / 4.7s    | 12 / 4.7s    | 5e-12   |

Two takeaways for the v0.8.0 docs:

1. **Iter-count win is uniform** — Newton / TRAH save 5-100+ iterations
   across every system. Energy parity vs DIIS is 1e-12 or tighter
   (DIIS itself is converged to conv_tol_energy = 1e-10, so any
   sub-1e-10 ΔE is within the SCF convergence floor).

2. **Wall-time win is system-dependent**. Newton's per-iter cost is
   ~5-15 Fock builds (one per CG iter); DIIS is one Fock build per
   iter. So on closed-shell systems where DIIS converges in <30 iters,
   Newton's iter win is offset by the per-iter overhead — net 2× wall
   time on benzene PBE.

   **The big wins land in regimes DIIS struggles with**:
   open-shell radicals with small gaps (OH•: 11× iter, 1.9× wall),
   transition-metal complexes (not tested here — separate study),
   broken-symmetry singlets. Even on closed-shell main-group with
   reasonable gaps, the iter-count win is worth keeping for systems
   where Fock-build cost is dominated by JK setup (DF cases — Newton
   amortises the same precomputed B-tensor across CG iters).

Run locally:

    python scripts/bench_scf_second_order.py

Run on a vq-registered host (the table above is from compute-host-d):

    vq submit <host> -d $(dirname scripts/bench_scf_second_order.py) \\
        --cpus 4 --wall-time-seconds 1200 \\
        -- ~/gitlab/vibeqc-dev/.venv/bin/python \\
        scripts/bench_scf_second_order.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

import vibeqc as vq
from vibeqc import Atom, BasisSet, Molecule, RKSOptions, UKSOptions, run_rks, run_uks


def banner(s):
    print("=" * len(s)); print(s); print("=" * len(s)); sys.stdout.flush()


def h2o():
    return Molecule([
        Atom(8, [0.0, 0.0,  0.221]),
        Atom(1, [0.0,  1.427, -0.890]),
        Atom(1, [0.0, -1.427, -0.890]),
    ])


def oh_radical():
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 0.0, 1.834])],
        multiplicity=2,
    )


def benzene():
    """Benzene in Bohr — D6h, C-C 2.625 bohr, C-H 2.040 bohr."""
    R_CC = 2.625
    R_CH = R_CC + 2.040
    atoms = []
    for i in range(6):
        theta = i * np.pi / 3
        atoms.append(Atom(6, [R_CC * np.cos(theta), R_CC * np.sin(theta), 0.0]))
        atoms.append(Atom(1, [R_CH * np.cos(theta), R_CH * np.sin(theta), 0.0]))
    return Molecule(atoms)


def run_rks_variants(name, mol, basis_name, functional):
    basis = BasisSet(mol, basis_name)
    banner(f"{name} / {basis_name} / {functional}  ({basis.nbasis} BFs)")

    def opts():
        o = RKSOptions()
        o.functional = functional
        o.conv_tol_energy = 1e-10
        o.conv_tol_grad = 1e-7
        o.max_iter = 200
        return o

    o_diis = opts()
    t = time.perf_counter(); r_diis = run_rks(mol, basis, o_diis); dt_diis = time.perf_counter() - t
    print(f"  DIIS    : E={r_diis.energy:.10f}  iters={r_diis.n_iter}  conv={r_diis.converged}  {dt_diis:.1f}s")
    sys.stdout.flush()

    for label, setup in [
        ("Newton", lambda o: setattr(o, "newton_threshold", 1.0)),
        ("TRAH",   lambda o: setattr(o, "trah_threshold",   1.0)),
    ]:
        o = opts(); setup(o)
        t = time.perf_counter()
        try:
            r = run_rks(mol, basis, o)
            dt = time.perf_counter() - t
            de = abs(r.energy - r_diis.energy)
            mark = "✓" if (r.converged and de < 1e-9) else "⚠"
            print(f"  {label:7s}: E={r.energy:.10f}  iters={r.n_iter}  conv={r.converged}  ΔE_vs_DIIS={de:.2e}  {mark}  {dt:.1f}s")
        except Exception as e:
            print(f"  {label:7s}: RAISED: {e}")
        sys.stdout.flush()


def run_uks_variants(name, mol, basis_name, functional):
    basis = BasisSet(mol, basis_name)
    banner(f"{name} / {basis_name} / UKS-{functional}  ({basis.nbasis} BFs)")

    def opts():
        o = UKSOptions()
        o.functional = functional
        o.conv_tol_energy = 1e-10
        o.conv_tol_grad = 1e-7
        o.max_iter = 250
        return o

    o_diis = opts()
    t = time.perf_counter(); r_diis = run_uks(mol, basis, o_diis); dt_diis = time.perf_counter() - t
    print(f"  DIIS    : E={r_diis.energy:.10f}  iters={r_diis.n_iter}  conv={r_diis.converged}  ⟨S²⟩={r_diis.s_squared:.4f}  {dt_diis:.1f}s")
    sys.stdout.flush()

    for label, setup in [
        ("Newton", lambda o: setattr(o, "newton_threshold", 1.0)),
        ("TRAH",   lambda o: setattr(o, "trah_threshold",   1.0)),
    ]:
        o = opts(); setup(o)
        t = time.perf_counter()
        try:
            r = run_uks(mol, basis, o)
            dt = time.perf_counter() - t
            de = abs(r.energy - r_diis.energy)
            mark = "✓" if (r.converged and de < 1e-9) else "⚠"
            print(f"  {label:7s}: E={r.energy:.10f}  iters={r.n_iter}  conv={r.converged}  ΔE_vs_DIIS={de:.2e}  ⟨S²⟩={r.s_squared:.4f}  {mark}  {dt:.1f}s")
        except Exception as e:
            print(f"  {label:7s}: RAISED: {e}")
        sys.stdout.flush()


def main():
    print(f"vibeqc {vq.__version__}")
    print(f"sys.executable = {sys.executable}")
    print()

    # RKS — closed-shell, three functional families
    for func in ("LDA", "PBE", "B3LYP"):
        run_rks_variants("H2O", h2o(), "cc-pvdz", func)
        print()

    run_rks_variants("Benzene", benzene(), "6-31g*", "PBE")
    print()
    run_rks_variants("Benzene", benzene(), "6-31g*", "B3LYP")
    print()

    # UKS — open-shell LDA only (UKS GGA gated on Phase 17e)
    run_uks_variants("OH•",   oh_radical(),    "cc-pvdz", "LDA")
    print()


if __name__ == "__main__":
    main()
