"""CASSCF analytic nuclear-gradient preview - diagnostic examples.

The current z-vector-free and optional W^z variants are not full-energy
finite-difference tight and must not be used as production forces. This file
exposes their components for development; the decisive correctness pin is
``examples/regression/casscf_gradient_fd_reproducer.py``.

Requires:
    pip install -e '.[test]'
    ./scripts/setup_native_deps.sh
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.gradient import compute_casscf_gradient
from vibeqc.molecular_optimize import optimize_molecule
from vibeqc.solvers import build_hamiltonian_mo, get_hf_orbital_provider
from vibeqc.solvers._casscf import CASSCFOptions, casscf
from vibeqc.solvers._rdm import make_rdm12, make_rdm12_sa


def example_h2_gradient():
    """H2/6-31G CAS(2,2) — single-point analytic gradient."""
    print("=" * 60)
    print("H2/6-31G CAS(2,2) n_core=0 — single-point gradient")
    print("=" * 60)

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "6-31g")

    # Run CASSCF
    C_hf = get_hf_orbital_provider(mol, basis)
    H = build_hamiltonian_mo(mol, basis, C_hf)
    sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
    assert sc.converged
    C_conv = C_hf @ sc.mo_rotation
    rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)

    # The analytic gradient: the complete derivative of the variational
    # CASSCF energy (matches full-energy FD to ~2e-7 Ha/bohr; GitLab #516).
    grad_87 = compute_casscf_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=0,
        n_active_orb=2,
        rdm1=rdm1,
        rdm2=rdm2,
    )
    print(f"  CASSCF energy:        {sc.e_total:.8f} Ha")
    print(
        f"  analytic gradient (atom 0): [{grad_87[0, 0]:.6f}, {grad_87[0, 1]:.6f}, {grad_87[0, 2]:.6f}]"
    )
    print(
        f"  Net force:            [{np.sum(grad_87[:, 0]):.2e}, {np.sum(grad_87[:, 1]):.2e}, {np.sum(grad_87[:, 2]):.2e}]"
    )

    # compute_wz="numerical": the in-repo finite-difference oracle.  The
    # former compute_wz=True "W^z correction" was retired (#516): a
    # variational CASSCF has no response term, and the experimental
    # implementation produced spurious gradient components.
    grad_fd = compute_casscf_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=0,
        n_active_orb=2,
        rdm1=rdm1,
        rdm2=rdm2,
        compute_wz="numerical",
    )
    print(
        f"  FD oracle (atom 0):         [{grad_fd[0, 0]:.6f}, {grad_fd[0, 1]:.6f}, {grad_fd[0, 2]:.6f}]"
    )
    print(
        f"  analytic - FD:              [{grad_87[0, 0] - grad_fd[0, 0]:.2e}, {grad_87[0, 1] - grad_fd[0, 1]:.2e}, {grad_87[0, 2] - grad_fd[0, 2]:.2e}]"
    )
    print()


def example_h2o_sa_gradient():
    """H2O/STO-3G SA2-CAS(4,4) — state-averaged gradient."""
    print("=" * 60)
    print("H2O/STO-3G SA2-CAS(4,4) n_core=1 — SA gradient")
    print("=" * 60)

    mol = Molecule(
        [
            Atom(8, [0, 0, 0.117]),
            Atom(1, [0, 0.757, -0.469]),
            Atom(1, [0, -0.757, -0.469]),
        ]
    )
    basis = BasisSet(mol, "sto-3g")

    C_hf = get_hf_orbital_provider(mol, basis)
    H = build_hamiltonian_mo(mol, basis, C_hf)
    sc = casscf(
        H.h1e,
        H.h2e,
        4,
        4,
        n_core=1,
        nuclear_repulsion=H.nuclear_repulsion,
        nroots=2,
        weights=[0.5, 0.5],
        max_macro=50,
    )
    assert sc.converged
    C_conv = C_hf @ sc.mo_rotation

    # State-averaged RDMs
    rdm1_sa, rdm2_sa = make_rdm12_sa(
        sc.cas.ci_coeffs_all, sc.cas.determinants, 4, [0.5, 0.5]
    )

    grad_sa = compute_casscf_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=1,
        n_active_orb=4,
        rdm1=rdm1_sa,
        rdm2=rdm2_sa,
    )
    print(f"  SA-CASSCF energy:      {sc.e_total:.8f} Ha")
    print(f"  Per-root energies:      {[f'{e:.8f}' for e in sc.e_totals]}")
    print(
        f"  SA gradient (atom 0):   [{grad_sa[0, 0]:.6f}, {grad_sa[0, 1]:.6f}, {grad_sa[0, 2]:.6f}]"
    )
    print(
        f"  Net force:              [{np.sum(grad_sa[:, 0]):.2e}, {np.sum(grad_sa[:, 1]):.2e}, {np.sum(grad_sa[:, 2]):.2e}]"
    )
    print()


def example_h2_optimization():
    """H2/6-31G CAS(2,2) — geometry optimization with analytic gradient."""
    print("=" * 60)
    print("H2/6-31G CAS(2,2) — geometry optimization")
    print("=" * 60)

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.5])])

    result = optimize_molecule(
        mol,
        "6-31g",
        method="casscf",
        active_space=(2, 2),
        casscf_options=CASSCFOptions(),
        max_iter=20,
        conv_tol_grad=1e-2,
        progress=True,
    )
    final_r = result.trajectory_frames[-1].atoms[1].xyz[2]
    print(f"  Starting R:    1.500 bohr")
    print(f"  Final R:       {final_r:.3f} bohr")
    print(f"  Final energy:  {result.energy:.8f} Ha")
    print(f"  Iterations:    {result.n_iter}")
    print(f"  Converged:     {result.converged}")
    print()


def example_h2o_optimization():
    """H2O/STO-3G CAS(4,4) — geometry optimization with analytic gradient."""
    print("=" * 60)
    print("H2O/STO-3G CAS(4,4) — geometry optimization")
    print("=" * 60)

    mol = Molecule(
        [
            Atom(8, [0, 0, 0.117]),
            Atom(1, [0, 1.0, -0.469]),  # stretched
            Atom(1, [0, -1.0, -0.469]),  # stretched
        ]
    )

    result = optimize_molecule(
        mol,
        "sto-3g",
        method="casscf",
        active_space=(4, 4),
        casscf_options=CASSCFOptions(),
        max_iter=30,
        conv_tol_grad=1e-2,
        progress=True,
    )
    print(f"  Final energy:  {result.energy:.8f} Ha")
    print(f"  Iterations:    {result.n_iter}")
    print()


def example_caspt2_gradient():
    """H2/6-31G CAS(2,2) — CASPT2 Z-vector gradient."""
    print("=" * 60)
    print("H2/6-31G CAS(2,2) — CASPT2 Z-vector gradient (99.99%)")
    print("=" * 60)
    from vibeqc.gradient._caspt2 import compute_caspt2_gradient

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "6-31g")
    C_hf = get_hf_orbital_provider(mol, basis)
    H = build_hamiltonian_mo(mol, basis, C_hf)
    sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
    C_conv = C_hf @ sc.mo_rotation
    rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)

    g = compute_caspt2_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=0,
        n_active_orb=2,
        n_active_elec=2,
        rdm1=rdm1,
        rdm2=rdm2,
        use_zvector=True,
        determinants=sc.cas.determinants,
        ci_coeffs=sc.cas.ci_coeffs,
    )
    print(f"  CASSCF energy:  {sc.e_total:.8f} Ha")
    print(f"  CASPT2 gradient: [{g[0, 0]:.6f}, {g[0, 1]:.6f}, {g[0, 2]:.6f}] Ha/bohr")
    print()


def example_nevpt2_gradient():
    """H2/6-31G CAS(2,2) — NEVPT2 Z-vector gradient."""
    print("=" * 60)
    print("H2/6-31G CAS(2,2) — NEVPT2 Z-vector gradient (99.99%)")
    print("=" * 60)
    from vibeqc.gradient._nevpt2 import compute_nevpt2_gradient

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "6-31g")
    C_hf = get_hf_orbital_provider(mol, basis)
    H = build_hamiltonian_mo(mol, basis, C_hf)
    sc = casscf(H.h1e, H.h2e, 2, 2, n_core=0, nuclear_repulsion=H.nuclear_repulsion)
    C_conv = C_hf @ sc.mo_rotation
    rdm1, rdm2 = make_rdm12(sc.cas.ci_coeffs, sc.cas.determinants, 2)

    g = compute_nevpt2_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=0,
        n_active_orb=2,
        n_active_elec=2,
        rdm1=rdm1,
        rdm2=rdm2,
        use_zvector=True,
        determinants=sc.cas.determinants,
        ci_coeffs=sc.cas.ci_coeffs,
    )
    print(f"  CASSCF energy:  {sc.e_total:.8f} Ha")
    print(f"  NEVPT2 gradient: [{g[0, 0]:.6f}, {g[0, 1]:.6f}, {g[0, 2]:.6f}] Ha/bohr")
    print()


def example_sa_nevpt2_gradient():
    """H2/6-31G SA-2-CAS(2,2) — SA-NEVPT2 Z-vector gradient (v52)."""
    print("=" * 60)
    print("H2/6-31G SA-2-CAS(2,2) — SA-NEVPT2 Z-vector gradient")
    print("=" * 60)
    from vibeqc.gradient._nevpt2 import compute_nevpt2_gradient

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "6-31g")
    C_hf = get_hf_orbital_provider(mol, basis)
    H = build_hamiltonian_mo(mol, basis, C_hf)
    sc = casscf(
        H.h1e,
        H.h2e,
        2,
        2,
        n_core=0,
        nuclear_repulsion=H.nuclear_repulsion,
        nroots=2,
        weights=[0.5, 0.5],
    )
    C_conv = C_hf @ sc.mo_rotation
    ci_all = sc.cas.ci_coeffs_all
    ci_list = [ci_all[:, i] for i in range(ci_all.shape[1])]
    rdm1_sa, rdm2_sa = make_rdm12_sa(ci_all, sc.cas.determinants, 2, [0.5, 0.5])

    g_sa = compute_nevpt2_gradient(
        mol,
        basis,
        C_conv,
        sc.h1e_cas,
        sc.h2e_cas,
        n_core=0,
        n_active_orb=2,
        n_active_elec=2,
        rdm1=rdm1_sa,
        rdm2=rdm2_sa,
        use_zvector=True,
        determinants=sc.cas.determinants,
        ci_coeffs=ci_list[0],
        sa_weights=[0.5, 0.5],
        sa_ci_coeffs=ci_list,
    )
    print(f"  SA-CASSCF energy: {sc.e_total:.8f} Ha")
    print(
        f"  SA-NEVPT2 grad:   [{g_sa[0, 0]:.6f}, {g_sa[0, 1]:.6f}, {g_sa[0, 2]:.6f}] Ha/bohr"
    )
    print()


# ── run examples ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_h2_gradient()
    example_h2o_sa_gradient()
    example_h2_optimization()
    # example_h2o_optimization()  # uncomment for heavier test (~30 iters)
    example_caspt2_gradient()
    example_nevpt2_gradient()
    example_sa_nevpt2_gradient()
    print("All examples complete.")
