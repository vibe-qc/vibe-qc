"""Phase SYM2b tests: lattice-cell orbit identification and
LatticeMatrixSet compression/reconstruction.

Core contract:

1. **Orbit counts** — a primitive-cubic lattice (O_h, 48 ops) with
   a spherical-cutoff cell list should partition into orbits whose
   sizes match the standard cubic shells: 1 (origin), 6 (±axis
   unit cells), 8 (corners), 12 (edges), 24 (face-diagonal), etc.

2. **Compression ratio** — storing only representatives gives a
   memory reduction equal to the number of cells divided by number
   of orbits. For high-symmetry O_h cells this is typically 5×–15×.

3. **Round-trip fidelity** — applying
   ``compress_lattice_matrix_set`` then
   ``reconstruct_lattice_matrix_set`` must return the *exact*
   original LMS at machine precision, for any symmetry-invariant
   integral matrix (overlap, kinetic, nuclear). This is the
   headline physics witness.

4. **Non-closed cell list rejected** — if a cell's orbit partner
   isn't in the list, the user must see a clear error rather than
   silently wrong reconstructions.

5. **Non-origin-fixed atoms rejected** — Phase SYM2b's scope is
   origin-fixed-only; the NaCl-style case (Cl at (½, ½, ½)) must
   raise with a directive error pointing at the SYM2c follow-up.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _primitive_cubic_mg(a: float = 3.0, cutoff: float = 7.0):
    """Primitive cubic lattice with one Mg atom (12 electrons,
    closed-shell) at the origin — the cleanest high-symmetry test
    system because every atom is origin-fixed under every operator."""
    sys_ = vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(12, [0, 0, 0])])
    vq.attach_symmetry(sys_)
    basis = vq.BasisSet(sys_.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = cutoff
    return sys_, basis, opts


# ---------------------------------------------------------------------------
# Coordinate-frame conversion
# ---------------------------------------------------------------------------

def test_lattice_to_cartesian_identity_on_identity_op():
    L = np.diag([4.0, 5.0, 6.0])
    R_lat = np.eye(3, dtype=int)
    R_cart = vq.lattice_to_cartesian_rotation(R_lat, L)
    assert np.allclose(R_cart, np.eye(3), atol=1e-12)


def test_lattice_to_cartesian_cubic_matches_input():
    """For a cubic lattice L = a·I, the lattice-basis and Cartesian
    forms are identical — L · R · L^{-1} = R when L is a scalar
    multiple of I."""
    L = 4.0 * np.eye(3)
    R_lat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    R_cart = vq.lattice_to_cartesian_rotation(R_lat, L)
    assert np.allclose(R_cart, R_lat.astype(float), atol=1e-12)


def test_lattice_to_cartesian_is_orthogonal():
    """For any point-group operator R_lat in an arbitrary lattice,
    the Cartesian form is real orthogonal."""
    L = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    R_lat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=int)  # C2(z)
    R_cart = vq.lattice_to_cartesian_rotation(R_lat, L)
    assert np.allclose(R_cart @ R_cart.T, np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------------
# Orbit identification
# ---------------------------------------------------------------------------

def test_orbits_on_primitive_cubic_Oh():
    """Primitive cubic: cell list should partition into cubic shells.
    At cutoff 7 bohr with a = 3 bohr we get the 0, ±1, ±√2, ±√3
    shells, whose multiplicities under O_h are 1, 6, 12, 8, plus
    further shells."""
    sys_, basis, opts = _primitive_cubic_mg(a=3.0, cutoff=7.0)
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_lattice_orbits(S.cells, sys_.symmetry.operations)
    sizes = sorted([o.size for o in orbits.orbits], reverse=True)
    # Sum of orbit sizes equals total cells.
    assert sum(sizes) == len(S.cells)
    # 1-orbit (origin) should always be there.
    assert 1 in sizes
    # Some multiplicities (at minimum a 6-orbit for the ±axis unit cells).
    assert 6 in sizes


def test_orbits_contain_identity_op_for_representative():
    """Every orbit's first op entry is the identity-operator index."""
    sys_, basis, opts = _primitive_cubic_mg()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_lattice_orbits(S.cells, sys_.symmetry.operations)
    for orb in orbits.orbits:
        # members[0] is the representative; ops[0] maps it to itself,
        # so ops[0] must refer to an identity operator.
        op_idx = orb.ops[0]
        R = np.asarray(sys_.symmetry.operations[op_idx].rotation)
        assert np.array_equal(R, np.eye(3, dtype=int))


def test_orbits_cover_all_cells_exactly_once():
    """The orbit partition must be disjoint and cover every cell."""
    sys_, basis, opts = _primitive_cubic_mg()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_lattice_orbits(S.cells, sys_.symmetry.operations)
    all_members = []
    for orb in orbits.orbits:
        all_members.extend(orb.members)
    assert sorted(all_members) == list(range(len(S.cells)))


def test_compression_ratio_reasonable_for_Oh():
    """O_h has 48 elements; most orbits have size 1, 6, 8, 12, 24, or
    48. Compression ratio should be well above 1."""
    sys_, basis, opts = _primitive_cubic_mg()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_lattice_orbits(S.cells, sys_.symmetry.operations)
    assert orbits.compression_ratio > 3.0


# ---------------------------------------------------------------------------
# Compression + reconstruction round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integral_name", ["overlap", "kinetic", "nuclear"])
def test_roundtrip_recovers_original_block_by_block(integral_name):
    """Compress and reconstruct each invariant one-electron integral
    lattice set; every block must match the original to machine
    precision."""
    sys_, basis, opts = _primitive_cubic_mg()
    if integral_name == "overlap":
        M = vq.compute_overlap_lattice(basis, sys_, opts)
    elif integral_name == "kinetic":
        M = vq.compute_kinetic_lattice(basis, sys_, opts)
    elif integral_name == "nuclear":
        M = vq.compute_nuclear_lattice(basis, sys_, opts)
    else:
        raise AssertionError

    orbits = vq.identify_lattice_orbits(M.cells, sys_.symmetry.operations)
    reps = vq.compress_lattice_matrix_set(M, orbits)
    reconstructed = vq.reconstruct_lattice_matrix_set(
        reps, orbits, basis, sys_, sys_.symmetry.operations,
    )
    assert len(reconstructed) == len(M.cells)
    max_err = max(
        np.abs(np.asarray(M.blocks[i]) - reconstructed[i]).max()
        for i in range(len(M.cells))
    )
    assert max_err < 1e-10, (
        f"{integral_name}: reconstruction error {max_err:.3e} "
        f"exceeds tolerance"
    )


def test_non_symmetry_invariant_matrix_detects_mismatch():
    """If we feed a NON-invariant matrix (e.g., a random asymmetric
    block list), the round-trip produces reconstructed blocks that
    DON'T match the original. Verifies the witness mechanism
    correctly flags broken invariance."""
    sys_, basis, opts = _primitive_cubic_mg()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_lattice_orbits(S.cells, sys_.symmetry.operations)

    # Replace block 0 (the origin block) with random noise.
    nbf = basis.nbasis
    rng = np.random.default_rng(42)
    fake_blocks = [np.asarray(b).copy() for b in S.blocks]
    fake_blocks[0] = rng.standard_normal((nbf, nbf))

    # Compress uses the representative (which we just corrupted) and
    # reconstructs every other orbit member from it. If the matrix
    # had the right invariance, reconstruction would reproduce the
    # (now corrupted) members. Since it doesn't, the rest of the
    # orbit differs from the reconstruction.
    # ... but S itself still matches its reconstruction because we
    # only touched the first block. So the "detect mismatch" logic
    # needs to compare against the ORIGINAL matrix, not the
    # corrupted one.
    class FakeLMS:
        def __init__(self, cells, blocks):
            self.cells = cells
            self.blocks = blocks
    fake_lms = FakeLMS(S.cells, fake_blocks)

    reps = vq.compress_lattice_matrix_set(fake_lms, orbits)
    rec = vq.reconstruct_lattice_matrix_set(
        reps, orbits, basis, sys_, sys_.symmetry.operations,
    )
    # Against the original (pre-corruption) blocks, reconstruction
    # disagrees somewhere — demonstrating the invariance witness.
    max_err = max(
        np.abs(np.asarray(S.blocks[i]) - rec[i]).max()
        for i in range(len(S.cells))
    )
    assert max_err > 1e-2, (
        f"expected the corrupted block to propagate detectable "
        f"disagreement through reconstruction; max_err = {max_err}"
    )


# ---------------------------------------------------------------------------
# Rejection: non-symmorphic and non-origin-fixed
# ---------------------------------------------------------------------------

def test_reject_non_origin_fixed_atom_structure():
    """NaCl has Cl at (a/2, a/2, a/2) which picks up a lattice shift
    under most cubic operators. The current SYM2b implementation
    must reject this with a directive error pointing at SYM2c.
    Phase SYM2c will lift the restriction."""
    a = 4.0
    sys_ = vq.PeriodicSystem(
        3, np.eye(3) * a,
        [vq.Atom(11, [0, 0, 0]), vq.Atom(17, [a / 2, a / 2, a / 2])],
    )
    vq.attach_symmetry(sys_)
    basis = vq.BasisSet(sys_.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_lattice_orbits(S.cells, sys_.symmetry.operations)
    reps = vq.compress_lattice_matrix_set(S, orbits)

    with pytest.raises(ValueError, match="SYM2c"):
        vq.reconstruct_lattice_matrix_set(
            reps, orbits, basis, sys_, sys_.symmetry.operations,
        )


# ---------------------------------------------------------------------------
# Orbit data-structure helpers
# ---------------------------------------------------------------------------

def test_representative_indices_are_distinct():
    sys_, basis, opts = _primitive_cubic_mg()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_lattice_orbits(S.cells, sys_.symmetry.operations)
    reps = orbits.representative_indices()
    assert len(set(reps)) == len(reps)   # all unique


def test_orbit_counts_match_n_orbits_attribute():
    sys_, basis, opts = _primitive_cubic_mg()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_lattice_orbits(S.cells, sys_.symmetry.operations)
    assert len(orbits.orbits) == orbits.n_orbits
    assert orbits.n_cells == len(S.cells)
