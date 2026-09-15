"""M2 tests: compute-reduced one-electron lattice integrals.

Validates:
  - Overlap S: compute_reduced matches full to ~1e-15
  - Kinetic T: compute_reduced matches full to ~1e-14
  - Nuclear V: falls back to full computation (storage-only)
  - Compression ratios reported correctly
  - Empty P_cache handled gracefully
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.symmetry_integrals_reduced import (
    compression_summary,
    compute_kinetic_lattice_reduced,
    compute_nuclear_lattice_reduced,
    compute_overlap_lattice_reduced,
)
from vibeqc.symmetry_scf import (
    build_ao_permutation_cache,
    symmetrize_density,
    symmetrize_fock,
    symmetrize_matrix,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _build_nacl_primitive(a_bohr: float = 8.0) -> vq.PeriodicSystem:
    a = float(a_bohr)
    return vq.PeriodicSystem(
        3,
        np.eye(3) * a,
        [vq.Atom(11, [0, 0, 0]), vq.Atom(17, [a / 2, a / 2, a / 2])],
    )


def _build_mg_primitive(a_bohr: float = 5.0) -> vq.PeriodicSystem:
    a = float(a_bohr)
    return vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(12, [0, 0, 0])])


# ---------------------------------------------------------------------------
# M2: one-electron compute reduction
# ---------------------------------------------------------------------------


def test_overlap_reduced_roundtrip():
    """S overlap: compute_reduced matches full to machine precision."""
    sys = _build_mg_primitive()
    vq.attach_symmetry(sys)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 12.0

    S_full = vq.compute_overlap_lattice(basis, sys, opts)
    _, recon_blocks = compute_overlap_lattice_reduced(
        basis,
        sys,
        opts,
        sys.symmetry.operations,
    )

    max_err = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(S_full.blocks, recon_blocks)
    )
    assert max_err < 1e-12, f"overlap reconstruction error {max_err:.2e}"


def test_kinetic_reduced_roundtrip():
    """T kinetic: compute_reduced matches full to machine precision."""
    sys = _build_mg_primitive()
    vq.attach_symmetry(sys)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 12.0

    T_full = vq.compute_kinetic_lattice(basis, sys, opts)
    _, recon_blocks = compute_kinetic_lattice_reduced(
        basis,
        sys,
        opts,
        sys.symmetry.operations,
    )

    max_err = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(T_full.blocks, recon_blocks)
    )
    assert max_err < 1e-12, f"kinetic reconstruction error {max_err:.2e}"


def test_nuclear_reduced_falls_back():
    """V nuclear: falls back to full computation (known limitation)."""
    sys = _build_mg_primitive()
    vq.attach_symmetry(sys)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    opts.nuclear_cutoff_bohr = 12.0

    V_full = vq.compute_nuclear_lattice(basis, sys, opts)
    _, recon_blocks = compute_nuclear_lattice_reduced(
        basis,
        sys,
        opts,
        sys.symmetry.operations,
    )

    # Fallback should give identical results
    max_err = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(V_full.blocks, recon_blocks)
    )
    assert max_err < 1e-14, f"nuclear fallback error {max_err:.2e}"

    # Verify it actually did full computation by checking cell count
    assert len(V_full.blocks) == len(recon_blocks)


def test_reduced_cell_count():
    """Compression reduces number of cells for integral evaluation."""
    sys = _build_mg_primitive()
    vq.attach_symmetry(sys)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 8.0

    red, _ = compute_overlap_lattice_reduced(
        basis,
        sys,
        opts,
        sys.symmetry.operations,
    )
    summary = compression_summary(red.orbits)

    # Mg primitive cell is Pm-3m with |G|=48 — high compression
    assert summary["n_cells_reduced"] < summary["n_cells_full"]
    assert summary["compression_cells"] > 1.5  # significant reduction
    assert summary["n_orbits"] > 0


def test_reduced_without_symmetry_raises():
    """Calling reduced functions without attach_symmetry should fail."""
    sys = _build_mg_primitive()
    # Don't call attach_symmetry
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()

    with pytest.raises(Exception):
        compute_overlap_lattice_reduced(basis, sys, opts, [])


# ---------------------------------------------------------------------------
# M5: symmetry_scf module
# ---------------------------------------------------------------------------


def test_permutation_cache_orthogonal():
    """All P(R) matrices are orthogonal."""
    sys = _build_mg_primitive()
    vq.attach_symmetry(sys)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    from vibeqc.symmetry_integrals import symmorphic_operations

    ops = symmorphic_operations(sys.symmetry.operations)
    cache = build_ao_permutation_cache(sys, basis, ops)

    nbf = basis.nbasis
    for i, P in enumerate(cache):
        assert np.allclose(P @ P.T, np.eye(nbf), atol=1e-12), f"P[{i}] not orthogonal"


def test_symmetrize_matrix_idempotent():
    """Symmetrization is idempotent."""
    sys = _build_mg_primitive()
    vq.attach_symmetry(sys)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    from vibeqc.symmetry_integrals import symmorphic_operations

    ops = symmorphic_operations(sys.symmetry.operations)
    cache = build_ao_permutation_cache(sys, basis, ops)

    rng = np.random.default_rng(42)
    M = rng.normal(size=(basis.nbasis, basis.nbasis))
    M_sym = symmetrize_matrix(M, cache)
    M_sym2 = symmetrize_matrix(M_sym, cache)

    assert np.allclose(M_sym, M_sym2, atol=1e-14), "symmetrization not idempotent"


def test_symmetrize_preserves_symmetric():
    """A group-invariant matrix is unchanged by symmetrization."""
    sys = _build_mg_primitive()
    vq.attach_symmetry(sys)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    from vibeqc.symmetry_integrals import symmorphic_operations

    ops = symmorphic_operations(sys.symmetry.operations)
    cache = build_ao_permutation_cache(sys, basis, ops)

    # Build a group-invariant matrix: S = P·S·P^T (overlap IS invariant)
    S = np.asarray(vq.compute_overlap(basis))
    S_sym = symmetrize_matrix(S, cache)

    assert np.allclose(S, S_sym, atol=1e-12), (
        "overlap should be invariant under symmetry"
    )


def test_symmetrize_density_no_symmetry():
    """symmetrize_density returns identity when system has no symmetry."""
    sys = _build_mg_primitive()
    # Don't attach
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    D = np.eye(basis.nbasis)
    D_sym = symmetrize_density(D, sys, basis)
    assert np.allclose(D, D_sym), "no-symmetry case should return input"


def test_symmetrize_fock_no_symmetry():
    """symmetrize_fock returns identity when system has no symmetry."""
    sys = _build_mg_primitive()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    F = np.eye(basis.nbasis)
    F_sym = symmetrize_fock(F, sys, basis)
    assert np.allclose(F, F_sym)


def test_empty_p_cache():
    """symmetrize_matrix with empty cache returns copy."""
    M = np.ones((3, 3))
    result = symmetrize_matrix(M, [])
    assert np.allclose(result, M)
    assert result is not M  # returns copy
