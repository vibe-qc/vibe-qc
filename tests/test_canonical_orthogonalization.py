"""Tests for canonical orthogonalisation in the SCF drivers.

Canonical orthogonalisation replaces the plain S^{-½} symmetric
orthogonaliser with a rectangular X (n_basis × n_kept) that projects
out near-null eigenvectors of the overlap matrix before the Fock
diagonalisation. Contract:

1. **Well-conditioned bases are unaffected.** The SCF energy,
   converged density and orbital energies on STO-3G / 6-31G* are
   identical (to 1e-10 Ha) across a range of ``linear_dep_threshold``
   values because no eigenvalue is below threshold anyway.

2. **Near-linearly-dependent bases converge instead of crashing.**
   The previously-pathological "tight H2 at 0.5 bohr with
   aug-cc-pVTZ" case (min S eig ≈ 1.9e-7) now runs through every SCF
   driver cleanly and returns a finite, physically-sensible energy.

3. **Dropping happens when the threshold says so.** Stricter
   thresholds drop strictly more basis functions; looser thresholds
   drop the same or fewer. The SCF energy is stable across the
   spread because the dropped directions are numerically decoupled
   from the physics.

4. **MO coefficient matrix shrinks accordingly.** ``mo_coeffs`` has
   shape ``(n_basis, n_kept)`` where ``n_kept <= n_basis``. When
   nothing is dropped they match; otherwise ``n_kept < n_basis``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Well-conditioned bases: linear_dep_threshold has no effect
# ---------------------------------------------------------------------------

def _h2o_molecule():
    return vq.Molecule([
        vq.Atom(8, [0, 0, 0]),
        vq.Atom(1, [0, 1.5, -1.2]),
        vq.Atom(1, [0, -1.5, -1.2]),
    ])


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g*"])
def test_well_conditioned_energy_invariant_to_threshold_rhf(basis_name):
    mol = _h2o_molecule()
    basis = vq.BasisSet(mol, basis_name)

    # Default threshold.
    r_default = vq.run_rhf(mol, basis)
    # Very loose threshold (should still drop nothing — eigenvalues
    # on these small clean bases are O(0.1) or larger).
    opts_loose = vq.RHFOptions()
    opts_loose.linear_dep_threshold = 1e-5
    r_loose = vq.run_rhf(mol, basis, opts_loose)
    # Zero threshold (disable canonical projection entirely).
    opts_zero = vq.RHFOptions()
    opts_zero.linear_dep_threshold = 0.0
    r_zero = vq.run_rhf(mol, basis, opts_zero)

    assert r_default.energy == pytest.approx(r_loose.energy, abs=1e-10)
    assert r_default.energy == pytest.approx(r_zero.energy, abs=1e-10)
    # On well-conditioned bases nothing is dropped.
    assert r_default.mo_coeffs.shape[1] == basis.nbasis
    assert r_loose.mo_coeffs.shape[1] == basis.nbasis


def test_well_conditioned_energy_invariant_rks():
    mol = _h2o_molecule()
    basis = vq.BasisSet(mol, "6-31g*")
    opts = vq.RKSOptions()
    opts.functional = "pbe"
    r_default = vq.run_rks(mol, basis, opts)
    opts.linear_dep_threshold = 1e-5
    r_loose = vq.run_rks(mol, basis, opts)
    assert r_default.energy == pytest.approx(r_loose.energy, abs=1e-10)


# ---------------------------------------------------------------------------
# Near-linearly-dependent basis: must not crash
# ---------------------------------------------------------------------------

def _tight_h2():
    return vq.Molecule([
        vq.Atom(1, [0, 0, 0]),
        vq.Atom(1, [0.5, 0, 0]),    # 0.5 bohr — absurdly tight
    ])


def test_tight_H2_augccpvtz_converges_rhf():
    """The pathological case from tests/test_linear_dependence.py: H2 at
    0.5 bohr with aug-cc-pVTZ. Before canonical orthogonalisation this
    threw ``RuntimeError: RHF: AO basis is linearly dependent ...``;
    now it should converge cleanly."""
    mol = _tight_h2()
    basis = vq.BasisSet(mol, "aug-cc-pvtz")
    # Force the projection to activate by tightening the threshold
    # above the known min eigenvalue ~1.9e-7.
    opts = vq.RHFOptions()
    opts.linear_dep_threshold = 1e-6
    r = vq.run_rhf(mol, basis, opts)
    assert r.converged
    # Energy must be finite — the value itself is physically silly
    # for 0.5-bohr H2 (far inside the nuclear-repulsion repulsion
    # barrier), but numerically it must be well-defined.
    assert np.isfinite(r.energy)
    # Basis was 46-dimensional; at least one direction dropped.
    assert r.mo_coeffs.shape[1] < basis.nbasis
    assert r.mo_coeffs.shape[0] == basis.nbasis


def test_tight_H2_augccpvtz_converges_rks():
    """Same pathological case, KS-DFT path."""
    mol = _tight_h2()
    basis = vq.BasisSet(mol, "aug-cc-pvtz")
    opts = vq.RKSOptions()
    opts.functional = "pbe"
    opts.linear_dep_threshold = 1e-6
    r = vq.run_rks(mol, basis, opts)
    assert r.converged
    assert np.isfinite(r.energy)
    assert r.mo_coeffs.shape[1] < basis.nbasis


# ---------------------------------------------------------------------------
# Dimension bookkeeping
# ---------------------------------------------------------------------------

def test_threshold_monotonicity_in_dropped_count():
    """Stricter thresholds drop at least as many directions as looser
    ones. Test on the tight-H2 case where exactly one direction hovers
    near 1.9e-7."""
    mol = _tight_h2()
    basis = vq.BasisSet(mol, "aug-cc-pvtz")
    n_bf = basis.nbasis
    drops = []
    for thresh in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        opts = vq.RHFOptions()
        opts.linear_dep_threshold = thresh
        r = vq.run_rhf(mol, basis, opts)
        drops.append(n_bf - r.mo_coeffs.shape[1])
    # drops should be non-decreasing as threshold increases.
    for i in range(len(drops) - 1):
        assert drops[i + 1] >= drops[i], (
            f"drops not monotone: {drops}"
        )


def test_energy_stable_across_thresholds_on_tight_case():
    """For the tight-H2 case, the total energy should be stable to
    ~1e-6 Ha across thresholds that all drop the same "bad" direction
    or leave it in (since its contribution to the wavefunction is
    numerically negligible by construction)."""
    mol = _tight_h2()
    basis = vq.BasisSet(mol, "aug-cc-pvtz")
    energies = []
    for thresh in (1e-7, 1e-6, 1e-5, 1e-4):
        opts = vq.RHFOptions()
        opts.linear_dep_threshold = thresh
        r = vq.run_rhf(mol, basis, opts)
        energies.append(r.energy)
    spread = max(energies) - min(energies)
    assert spread < 1e-6, f"energy spread across thresholds = {spread:.2e}"


# ---------------------------------------------------------------------------
# UHF / UKS paths
# ---------------------------------------------------------------------------

def test_canonical_orth_uhf_on_open_shell():
    """Open-shell case (H atom) with linear-dep threshold set — must
    converge. No near-null space here, just verifies the UHF path
    compiles and runs through canonical orth."""
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0])], charge=0, multiplicity=2)
    basis = vq.BasisSet(mol, "6-31g*")
    opts = vq.UHFOptions()
    r = vq.run_uhf(mol, basis, opts)
    assert r.converged
    # H atom in 6-31G*: a few basis functions, no dependence.
    assert r.mo_coeffs_alpha.shape[1] == basis.nbasis


def test_canonical_orth_uks_on_open_shell():
    """Same for UKS — just exercise the code path."""
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0])], charge=0, multiplicity=2)
    basis = vq.BasisSet(mol, "6-31g*")
    opts = vq.UKSOptions()
    opts.functional = "lda"
    r = vq.run_uks(mol, basis, opts)
    assert r.converged


# ---------------------------------------------------------------------------
# Error path: too many electrons for the projected space
# ---------------------------------------------------------------------------

def test_rejects_when_too_many_electrons_for_kept_dimension():
    """If the threshold is aggressive enough to drop more basis
    functions than there are occupied orbitals, the SCF must fail
    with a directive error message."""
    mol = _h2o_molecule()
    basis = vq.BasisSet(mol, "sto-3g")   # 7 AOs, 10 electrons → 5 occ
    opts = vq.RHFOptions()
    # Threshold higher than the largest S eigenvalue → drops EVERY basis
    # function, so n_kept = 0 < n_occ = 5.
    opts.linear_dep_threshold = 10.0
    with pytest.raises((RuntimeError, ValueError)):
        vq.run_rhf(mol, basis, opts)
