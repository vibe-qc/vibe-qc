"""Phase 12b: Γ-only periodic RHF in the molecular-limit regime.

The calculation is well-defined whenever the unit cell is large enough that
P_μν(g ≠ 0) is numerically zero — in that regime the periodic Fock build
reduces to the molecular Fock build and the total energy must match
molecular RHF to machine precision.

Bulk systems with genuine inter-cell bonding need multi-k sampling of the
real-space density matrix; that's Phase 12c.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _molecular_reference(atoms):
    mol = vq.Molecule(atoms, 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return vq.run_rhf(mol, basis, opts), mol, basis


def _periodic_gamma(atoms, dim, lattice, basis_name="sto-3g",
                    box_side=50.0, cutoff_bohr=15.0, *,
                    molecular_limit=False):
    sysp = vq.PeriodicSystem(dim, lattice, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), basis_name)
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff_bohr
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff_bohr
    # Issue #133: the 50-bohr-box callers below deliberately pick a cutoff that
    # leaves only g = 0 -- that IS the premise of their molecular-limit
    # assertions. run_rhf_periodic_gamma now refuses such a cell list unless
    # the caller declares it, so they declare it. It changes no sum.
    opts.lattice_opts.gamma_only_0 = molecular_limit
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return vq.run_rhf_periodic_gamma(sysp, basis, opts), sysp


# Standard geometries.
H2  = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
H2O = [vq.Atom(8, [0.0,  0.0,  0.0]),
       vq.Atom(1, [0.0,  1.43, -0.98]),
       vq.Atom(1, [0.0, -1.43, -0.98])]
CH4 = [vq.Atom(6, [ 0.0,  0.0,  0.0]),
       vq.Atom(1, [+1.19, +1.19, +1.19]),
       vq.Atom(1, [-1.19, -1.19, +1.19]),
       vq.Atom(1, [+1.19, -1.19, -1.19]),
       vq.Atom(1, [-1.19, +1.19, -1.19])]


MOL_LIMIT_CASES = [
    ("H2",  H2),
    ("H2O", H2O),
    ("CH4", CH4),
]


@pytest.mark.parametrize(
    "name,atoms,dim",
    [(n, a, d) for (n, a) in MOL_LIMIT_CASES for d in (1, 2, 3)],
    ids=[f"{n}-dim{d}" for (n, _) in MOL_LIMIT_CASES for d in (1, 2, 3)],
)
def test_molecular_limit_matches_molecular_rhf(name, atoms, dim):
    """Total energy matches molecular RHF to machine precision when the
    lattice-sum cutoff leaves only the g=0 cell in range."""
    box = 50.0
    if dim == 1:   lat = np.diag([box, 30.0, 30.0])
    elif dim == 2: lat = np.diag([box, box, 30.0])
    else:          lat = np.diag([box, box, box])

    mres, mol, _ = _molecular_reference(atoms)
    res, _ = _periodic_gamma(atoms, dim, lat, cutoff_bohr=15.0, molecular_limit=True)

    assert res.converged
    assert abs(res.energy - mres.energy) < 1e-10, (
        f"{name}/dim={dim}: periodic {res.energy:.12f} vs molecular "
        f"{mres.energy:.12f}, diff = {res.energy - mres.energy:+.2e}"
    )
    # MO eigenvalue spectrum must also match. Tolerance 1e-7 accommodates
    # diagonalisation noise between the two independent numerical paths;
    # the total energy above is already pinned at 1e-10.
    assert np.max(np.abs(np.asarray(res.mo_energies) -
                         np.asarray(mres.mo_energies))) < 1e-7


def test_molecular_limit_density_matches():
    """Not just the energy — the converged density matrix at Γ must match
    the molecular RHF density to machine precision."""
    mres, _, _ = _molecular_reference(H2O)
    res, _ = _periodic_gamma(H2O, 3, np.eye(3) * 50.0, cutoff_bohr=15.0, molecular_limit=True)
    D_p = np.asarray(res.density)
    D_m = np.asarray(mres.density)
    # Same 1e-7 tolerance rationale as the MO eigenvalues: different
    # numerical paths, same energy to 1e-10 but density elements land on
    # the same self-consistent fixed point to ~1e-8.
    assert np.max(np.abs(D_p - D_m)) < 1e-7


def test_lattice_sum_exercises_multiple_cells_and_converges():
    """Verify the lattice-sum loop is actually entered for N_cells > 1
    without breaking SCF convergence. Quantitative energy comparison
    against molecular is NOT asserted here — with neighbor image cells
    in the ERI sum, the SCF converges to a solution that includes those
    (in molecular limit negligible) contributions, and numerical
    accumulation over many shell quartets perturbs it at well past 1e-10
    precision. The clean quantitative check lives in
    ``test_molecular_limit_matches_molecular_rhf``.
    """
    sysp = vq.PeriodicSystem(1, np.diag([10.0, 30.0, 30.0]), H2)
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 25.0        # ±2 cells along 1D axis
    opts.lattice_opts.nuclear_cutoff_bohr = 50.0
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.max_iter = 100
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    res = vq.run_rhf_periodic_gamma(sysp, basis, opts)
    n_cells = len(vq.direct_lattice_cells(sysp, opts.lattice_opts.cutoff_bohr))
    assert n_cells >= 5, "test prerequisite: cutoff must include ≥ 5 cells"
    assert res.converged


def test_nuclear_repulsion_per_cell_matches_molecular():
    """When the cutoff includes only g=0, the periodic nuclear repulsion
    must equal the molecular nuclear repulsion."""
    sysp = vq.PeriodicSystem(3, np.eye(3) * 50.0, H2O)
    opts = vq.LatticeSumOptions()
    opts.nuclear_cutoff_bohr = 15.0
    e_periodic = vq.nuclear_repulsion_per_cell(sysp, opts)
    mol = vq.Molecule(H2O, 0, 1)
    e_mol = mol.nuclear_repulsion()
    assert abs(e_periodic - e_mol) < 1e-12


def test_rejects_odd_electron_unit_cell():
    """Open-shell unit cells must be rejected by the RHF driver."""
    # H atom alone is odd-electron; multiplicity=2 to make Molecule happy,
    # but run_rhf_periodic_gamma requires multiplicity=1.
    sysp = vq.PeriodicSystem(3, np.eye(3) * 10.0,
                             [vq.Atom(1, [0, 0, 0])],
                             charge=0, multiplicity=2)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="even electron count"):
        vq.run_rhf_periodic_gamma(sysp, basis)


def test_scf_trace_populated():
    res, _ = _periodic_gamma(H2, 3, np.eye(3) * 50.0, cutoff_bohr=15.0, molecular_limit=True)
    assert len(res.scf_trace) >= 2
    assert res.scf_trace[0].iter == 1
    # Convergence improves monotonically for this simple case.
    assert res.scf_trace[-1].grad_norm < res.scf_trace[0].grad_norm


def test_sad_and_hcore_guesses_converge_to_same_energy():
    """SAD and Hcore initial guesses must land on the same SCF minimum."""
    sysp = vq.PeriodicSystem(3, np.eye(3) * 50.0, H2O)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    def _run(guess):
        o = vq.PeriodicRHFOptions()
        o.lattice_opts.cutoff_bohr = 15.0
        o.lattice_opts.nuclear_cutoff_bohr = 15.0
        o.lattice_opts.gamma_only_0 = True  # 50-bohr box (#133)
        o.conv_tol_energy = 1e-12
        o.initial_guess = guess
        return vq.run_rhf_periodic_gamma(sysp, basis, o)

    e_hcore = _run(vq.InitialGuess.HCORE).energy
    e_sad   = _run(vq.InitialGuess.SAD).energy
    assert abs(e_hcore - e_sad) < 1e-10
