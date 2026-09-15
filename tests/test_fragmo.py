"""FRAGMO initial guess — fragment-superposition SCF guess (molecular, roadmap D2h).

FRAGMO converges each fragment on its own and assembles the densities
block-diagonally into the supersystem guess. The two things it must do:

* **Correctness (guess-independence):** the converged SCF energy must match
  the SAD/HCORE result — the guess only changes the path, not the minimum.
* **Value:** on weakly-interacting dimers the assembled guess starts close to
  the supersystem solution, so it converges in *fewer* SCF iterations than SAD.
"""
from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Fragment,
    InitialGuess,
    Molecule,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    run_rhf,
    run_rks,
    run_uhf,
)

# Water monomer (bohr): O at origin, two O–H ≈ 0.96 Å at the HOH angle.
_WATER = [
    (8, [0.000000, 0.000000, 0.000000]),
    (1, [0.000000, 1.431500, 1.107153]),
    (1, [0.000000, -1.431500, 1.107153]),
]


def _water_dimer(sep_bohr: float) -> Molecule:
    """Two waters stacked ``sep_bohr`` apart along +z (atoms 0–2 / 3–5)."""
    atoms = [Atom(z, list(xyz)) for z, xyz in _WATER]
    shift = np.array([0.0, 0.0, sep_bohr])
    atoms += [Atom(z, list(np.array(xyz) + shift)) for z, xyz in _WATER]
    return Molecule(atoms, charge=0, multiplicity=1)


def _rhf_opts(guess: InitialGuess) -> RHFOptions:
    o = RHFOptions()
    o.conv_tol_energy = 1e-11
    o.conv_tol_grad = 1e-9
    o.initial_guess = guess
    return o


# ---------------------------------------------------------------------------
# Correctness: FRAGMO converges to the same minimum as SAD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sep", [5.6, 8.0])
def test_fragmo_matches_sad_energy_water_dimer(sep):
    mol = _water_dimer(sep)
    basis = BasisSet(mol, "sto-3g")
    frags = [Fragment(atoms=[0, 1, 2]), Fragment(atoms=[3, 4, 5])]
    r_fragmo = run_rhf(mol, basis, _rhf_opts(InitialGuess.FRAGMO), fragments=frags)
    r_sad = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAD))
    assert r_fragmo.converged and r_sad.converged
    assert abs(r_fragmo.energy - r_sad.energy) < 1e-8


def test_fragmo_accepts_plain_atom_index_lists():
    """fragments=[[...], [...]] (no Fragment wrapper) → neutral closed-shell."""
    mol = _water_dimer(5.6)
    basis = BasisSet(mol, "sto-3g")
    r = run_rhf(mol, basis, _rhf_opts(InitialGuess.FRAGMO),
                fragments=[[0, 1, 2], [3, 4, 5]])
    r_sad = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAD))
    assert abs(r.energy - r_sad.energy) < 1e-8


# ---------------------------------------------------------------------------
# Value: FRAGMO beats SAD on iteration count for weakly-interacting dimers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sep", [6.5, 9.0])
def test_fragmo_beats_sad_iterations(sep):
    mol = _water_dimer(sep)
    basis = BasisSet(mol, "6-31g")
    frags = [Fragment(atoms=[0, 1, 2]), Fragment(atoms=[3, 4, 5])]
    r_fragmo = run_rhf(mol, basis, _rhf_opts(InitialGuess.FRAGMO), fragments=frags)
    r_sad = run_rhf(mol, basis, _rhf_opts(InitialGuess.SAD))
    assert abs(r_fragmo.energy - r_sad.energy) < 1e-8
    assert r_fragmo.n_iter < r_sad.n_iter


# ---------------------------------------------------------------------------
# RKS + open-shell driver paths
# ---------------------------------------------------------------------------


def test_fragmo_rks_matches_sad_energy():
    mol = _water_dimer(5.6)
    basis = BasisSet(mol, "sto-3g")
    frags = [Fragment(atoms=[0, 1, 2]), Fragment(atoms=[3, 4, 5])]

    def opts(guess):
        o = RKSOptions()
        o.functional = "pbe"
        o.conv_tol_energy = 1e-10
        o.conv_tol_grad = 1e-8
        o.initial_guess = guess
        return o

    r_fragmo = run_rks(mol, basis, opts(InitialGuess.FRAGMO), fragments=frags)
    r_sad = run_rks(mol, basis, opts(InitialGuess.SAD))
    assert r_fragmo.converged
    assert abs(r_fragmo.energy - r_sad.energy) < 1e-7


def test_fragmo_uhf_closed_shell_matches_rhf():
    """A closed-shell supersystem run through UHF/FRAGMO (each fragment's total
    density split ½/½ into α/β) must reproduce the RHF/FRAGMO energy."""
    mol = _water_dimer(5.6)
    basis = BasisSet(mol, "sto-3g")
    frags = [Fragment(atoms=[0, 1, 2]), Fragment(atoms=[3, 4, 5])]
    uo = UHFOptions()
    uo.conv_tol_energy = 1e-11
    uo.conv_tol_grad = 1e-8
    uo.initial_guess = InitialGuess.FRAGMO
    r_uhf = run_uhf(mol, basis, uo, fragments=frags)
    r_rhf = run_rhf(mol, basis, _rhf_opts(InitialGuess.FRAGMO), fragments=frags)
    assert r_uhf.converged
    assert abs(r_uhf.energy - r_rhf.energy) < 1e-7


# ---------------------------------------------------------------------------
# Honest errors — partition validation + regime / pairing guards
# ---------------------------------------------------------------------------


def test_fragmo_requires_fragments():
    mol = _water_dimer(5.6)
    basis = BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="FRAGMO requires fragments"):
        run_rhf(mol, basis, _rhf_opts(InitialGuess.FRAGMO))


def test_fragmo_rejects_incomplete_partition():
    mol = _water_dimer(5.6)
    basis = BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="cover every atom"):
        run_rhf(mol, basis, _rhf_opts(InitialGuess.FRAGMO),
                fragments=[[0, 1, 2], [3, 4]])  # atom 5 missing


def test_fragmo_rejects_overlapping_partition():
    mol = _water_dimer(5.6)
    basis = BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="disjoint partition"):
        run_rhf(mol, basis, _rhf_opts(InitialGuess.FRAGMO),
                fragments=[[0, 1, 2], [2, 3, 4, 5]])  # atom 2 in both


def test_fragmo_rejects_charge_mismatch():
    mol = _water_dimer(5.6)
    basis = BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="fragment charges sum"):
        run_rhf(mol, basis, _rhf_opts(InitialGuess.FRAGMO),
                fragments=[Fragment(atoms=[0, 1, 2], charge=1),
                           Fragment(atoms=[3, 4, 5], charge=0)])


def test_fragments_without_fragmo_guess_raises():
    mol = _water_dimer(5.6)
    basis = BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="not FRAGMO"):
        run_rhf(mol, basis, _rhf_opts(InitialGuess.SAD),
                fragments=[[0, 1, 2], [3, 4, 5]])


def test_fragmo_periodic_requires_fragment_context_at_python_edge():
    """The density-only helper cannot construct periodic fragment sources."""
    import vibeqc as vq
    from vibeqc.guess import initial_density_closed_shell
    lat = 30.0 * np.eye(3)
    sysp = vq.PeriodicSystem(3, lat, [vq.Atom(1, [0.0, 0.0, 0.0]),
                                      vq.Atom(1, [1.4, 0.0, 0.0])],
                             charge=0, multiplicity=1)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    mol = sysp.unit_cell_molecule()
    with pytest.raises(NotImplementedError, match="FRAGMO requires fragment atom/image ownership"):
        initial_density_closed_shell(mol, basis, 1, InitialGuess.FRAGMO,
                                     is_periodic=True)


def test_fragment_preserves_modified_parent_shells_and_atom_order():
    from vibeqc.guess_fragmo import _build_fragment
    from vibeqc import compute_overlap

    mol = Molecule([Atom(2, [0., 0., z]) for z in (0., 3., 7.)])
    shells = BasisSet(mol, "sto-3g").shells()
    for index, shell in enumerate(shells):
        shell.exponents = [(1.2 + index) * value for value in shell.exponents]
    parent = BasisSet(mol, shells, "sto-3g")
    _, fragment_basis, order = _build_fragment(mol, parent, Fragment([2, 0]))
    assert order == [2, 0]
    expected = np.asarray(compute_overlap(parent))[np.ix_(order, order)]
    np.testing.assert_allclose(compute_overlap(fragment_basis), expected, atol=1e-12)


def test_periodic_fragment_translates_the_actual_ecp_with_its_atom():
    import vibeqc as vq
    from vibeqc.periodic_runner import _resolve_ecp_data
    from vibeqc.guess_fragmo import resolve_periodic_fragmo_source

    system = vq.PeriodicSystem(3, np.eye(3) * 16,
                              [Atom(11, [0., 0., 16.]), Atom(1, [0., 0., 3.5])])
    basis = BasisSet(system.unit_cell_molecule(), "lanl2dz")
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy = 1e-11
    options.conv_tol_grad = 1e-9
    (options.ecp_primitive_blocks, options.ecp_home_centers,
     options.ecp_effective_charges, options.ecp_total_ncore) = _resolve_ecp_data(system, basis)
    mesh = vq.monkhorst_pack(system, [1, 1, 3])
    source = resolve_periodic_fragmo_source(options, system, basis, mesh, [
        vq.PeriodicFragment([0, 1], images=[[0, 0, -1], [0, 0, 0]])])
    molecule = Molecule([Atom(11, [0., 0., 0.]), Atom(1, [0., 0., 3.5])])
    reference_basis = BasisSet(molecule, "lanl2dz")
    reference = run_rhf(molecule, reference_basis, _rhf_opts(InitialGuess.SAD))
    assert reference.converged
    np.testing.assert_allclose(source.density[0], reference.density, atol=1e-8)
    assert np.trace(source.density[0] @ vq.compute_overlap(reference_basis)) == pytest.approx(2.)
    np.testing.assert_allclose(options.ecp_home_centers, [[0., 0., 16.]])


def test_periodic_fragment_spin_orientation_preserves_antiferromagnetic_seed():
    import vibeqc as vq
    from vibeqc.guess_fragmo import resolve_periodic_fragmo_source

    system = vq.PeriodicSystem(3, np.eye(3) * 16,
                              [Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 6.])])
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    source = resolve_periodic_fragmo_source(vq.PeriodicRHFOptions(), system, basis,
        vq.monkhorst_pack(system, [1, 1, 3]), [
            vq.PeriodicFragment([0], multiplicity=2),
            vq.PeriodicFragment([1], multiplicity=2, spin_orientation=-1)])
    for alpha, beta in zip(source.density_alpha, source.density_beta):
        np.testing.assert_allclose(alpha, np.diag([1., 0.]), atol=1e-12)
        np.testing.assert_allclose(beta, np.diag([0., 1.]), atol=1e-12)
