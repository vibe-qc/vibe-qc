"""End-to-end demo: drive a Γ-only periodic RHF SCF using the same
``JKBuilder`` interface that the molecular SCF drivers use.

The molecular SCF iteration body — canonical orthogonalisation, DIIS,
damping, optional level-shift, optional quadratic fallback,
post-convergence Fock rebuild — all lives behind one C++ entry
point: ``run_rhf_scf_with_jk(basis, n_e, S, Hcore, E_nuc, jk, opts,
initial_density=…)``. The molecular ``run_rhf`` is now a thin wrapper
that builds Hcore + E_nuc from the molecular integrals and a
JKBuilder from the density_fit / cosx options, then calls into the
same SCF body.

For periodic-Γ we just substitute the ingredients:

  * ``Hcore``    via lattice-summed T + V_ne (Γ-fold).
  * ``E_nuc``    via Ewald / direct-truncated nuclear repulsion per cell.
  * ``S``        via lattice-summed overlap (Γ-fold).
  * ``jk``       via ``make_periodic_gamma_jk_builder``.

Everything else — DIIS, level shift, quadratic fallback, the SCF
trace, the energy decomposition, the convergence check — comes from
the molecular code path with no further changes.
"""
import numpy as np

import vibeqc as vq
from vibeqc.periodic_gradient import _fold_gamma_real


def run_periodic_rhf_gamma(system, basis, *, n_electrons,
                           cutoff_bohr=25.0, options=None):
    """Closed-shell Γ-only periodic RHF SCF using the JKBuilder /
    SCF-with-jk abstraction. Returns an ``RHFResult`` — same shape
    and fields as a molecular ``run_rhf`` result (energy,
    mo_energies, mo_coeffs, density, fock, scf_trace, n_iter,
    converged)."""
    if options is None:
        options = vq.RHFOptions()

    lattice_opts = vq.LatticeSumOptions()
    lattice_opts.cutoff_bohr = cutoff_bohr

    # Periodic ingredients — Γ-folded one-electron + Ewald nuc-rep.
    S     = _fold_gamma_real(vq.compute_overlap_lattice(basis, system, lattice_opts))
    T     = _fold_gamma_real(vq.compute_kinetic_lattice(basis, system, lattice_opts))
    V_ne  = _fold_gamma_real(vq.compute_nuclear_lattice(basis, system, lattice_opts))
    Hcore = T + V_ne
    E_nuc = vq.nuclear_repulsion_per_cell(system, lattice_opts)

    # JKBuilder for the periodic-Γ Fock 2e piece.
    jk = vq.make_periodic_gamma_jk_builder(basis, system, lattice_opts)

    # Drive the molecular SCF body — DIIS / damping / level-shift /
    # quadratic fallback all come along for free.
    return vq.run_rhf_scf_with_jk(
        basis, n_electrons, S, Hcore, E_nuc, jk, options)


if __name__ == "__main__":
    # Tiny 1D H chain — 2 atoms / cell, lattice = 5 bohr along x.
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])]
    mol = vq.Molecule(atoms)
    basis = vq.BasisSet(mol, "sto-3g")
    lattice = np.array([[5.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    system = vq.PeriodicSystem(1, lattice, atoms)

    options = vq.RHFOptions()
    options.conv_tol_energy = 1e-9
    options.conv_tol_grad = 1e-7
    options.use_diis = True

    result = run_periodic_rhf_gamma(system, basis, n_electrons=2,
                                    options=options)

    print(f"H chain (1D, a = 5 bohr) / sto-3g  Γ-only RHF")
    print(f"  E_total      = {result.energy:.10f} Ha")
    print(f"  iters        = {result.n_iter}  (converged = {result.converged})")
    print(f"  occupied MOs = {np.array(result.mo_energies)[:1]}")
    print(f"  virtual  MOs = {np.array(result.mo_energies)[1:]}")
    print()
    print("  iteration trace:")
    for it in result.scf_trace:
        print(f"    iter {it.iter:2d}  E = {it.energy:.10f}  "
              f"||[F,DS]|| = {it.grad_norm:.3e}  "
              f"DIIS = {it.diis_subspace}")
