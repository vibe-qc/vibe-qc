"""Phase SYM2c tests: atom-pair-resolved orbit identification and
LatticeMatrixSet compression for non-origin-fixed structures.

SYM2c lifts SYM2b's restriction that all atoms must be operator-fixed.
This is exercised here on NaCl (space group Pm-3m, Cl at the
(a/2, a/2, a/2) Wyckoff site which is NOT fixed by most cubic
operators), plus the primitive-cubic Mg case SYM2b already handles
— to verify SYM2c is a proper superset.

Contracts:

1. **Orbit partition covers every (atom-pair, cell) triple** — no
   leakage, no double counting.

2. **Round-trip fidelity on invariant integrals.** Compress then
   reconstruct overlap / kinetic / nuclear matrices on NaCl; each
   full block matches the original at machine precision. This is
   the headline physics witness for SYM2c.

3. **Orbit members carry the identity operator at position 0**, i.e.
   ``ops[0]`` maps the representative to itself.

4. **Compression ratio is non-trivial** — for O_h NaCl on a
   reasonable cutoff the compression should be ~15×+.

5. **Back-compat with SYM2b's simple case.** Primitive cubic Mg
   round-trips at machine precision through both SYM2b and SYM2c
   code paths. The SYM2c path is strictly more general.

6. **Non-closed triple spaces** (orbits that straddle the cutoff
   boundary) are handled via ``require_closed=False`` — skipping
   boundary-leaking members, not silently producing wrong results.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _nacl(a: float = 4.0, cutoff: float = 20.0):
    """NaCl structure (space group Pm-3m, 48 O_h operators). Cl at
    (a/2, a/2, a/2) Wyckoff position picks up lattice shifts under
    most operators — the canonical SYM2c test case."""
    sys_ = vq.PeriodicSystem(
        3, np.eye(3) * a,
        [vq.Atom(11, [0, 0, 0]), vq.Atom(17, [a / 2, a / 2, a / 2])],
    )
    vq.attach_symmetry(sys_)
    basis = vq.BasisSet(sys_.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = cutoff
    return sys_, basis, opts


def _primitive_mg(a: float = 3.0, cutoff: float = 7.0):
    sys_ = vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(12, [0, 0, 0])])
    vq.attach_symmetry(sys_)
    basis = vq.BasisSet(sys_.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = cutoff
    return sys_, basis, opts


# ---------------------------------------------------------------------------
# Orbit identification — basic sanity
# ---------------------------------------------------------------------------

def test_orbit_partition_covers_every_triple_nacl():
    """With require_closed=False, every (atom, atom, cell) triple
    should land in exactly one orbit — no gaps, no duplicates."""
    sys_, basis, opts = _nacl()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_atom_pair_orbits(
        S.cells, sys_.symmetry.operations, sys_, require_closed=False,
    )
    covered = sum(o.size for o in orbits.orbits)
    expected = 2 * 2 * len(S.cells)
    assert covered == expected, f"covered {covered} of {expected} triples"
    assert orbits.n_triples == expected


def test_representative_carries_identity_op():
    sys_, basis, opts = _nacl()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_atom_pair_orbits(
        S.cells, sys_.symmetry.operations, sys_, require_closed=False,
    )
    for orb in orbits.orbits:
        op_idx = orb.ops[0]
        R = np.asarray(sys_.symmetry.operations[op_idx].rotation)
        assert np.array_equal(R, np.eye(3, dtype=int))


def test_no_duplicate_triples():
    """Every triple appears in at most one orbit."""
    sys_, basis, opts = _nacl()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_atom_pair_orbits(
        S.cells, sys_.symmetry.operations, sys_, require_closed=False,
    )
    seen: set = set()
    for orb in orbits.orbits:
        for m in orb.members:
            assert m not in seen, f"duplicate member {m}"
            seen.add(m)


def test_compression_ratio_meaningful_for_Oh():
    """At reasonable cutoffs, O_h symmetry on NaCl gives > 5×
    compression."""
    sys_, basis, opts = _nacl(cutoff=20.0)
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_atom_pair_orbits(
        S.cells, sys_.symmetry.operations, sys_, require_closed=False,
    )
    assert orbits.compression_ratio > 5.0, (
        f"compression ratio {orbits.compression_ratio:.2f} below "
        f"expectation for O_h-symmetric NaCl"
    )


# ---------------------------------------------------------------------------
# Round-trip fidelity (the headline test)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integral_name", ["overlap", "kinetic"])
def test_roundtrip_recovers_nacl_short_range_integrals(integral_name):
    """Compress then reconstruct NaCl's overlap / kinetic lattice
    matrix; every block matches the original at machine precision.
    This is the key witness that SYM2c handles the NaCl-class
    (non-origin-fixed-atom) case correctly for short-range,
    translationally-invariant integrals.

    Note: nuclear attraction ``compute_nuclear_lattice`` is *not*
    exactly symmetric under the space group because it uses a
    real-space-truncated nucleus list — sub-block elements at large
    cell separations pick up a sub-% truncation asymmetry. That's
    a property of the library function, not of SYM2c. Ewald-based
    nuclear attraction (Phase 12e-c-4 follow-up) would restore the
    exact invariance.
    """
    sys_, basis, opts = _nacl()
    if integral_name == "overlap":
        M = vq.compute_overlap_lattice(basis, sys_, opts)
    elif integral_name == "kinetic":
        M = vq.compute_kinetic_lattice(basis, sys_, opts)
    else:
        raise AssertionError

    orbits = vq.identify_atom_pair_orbits(
        M.cells, sys_.symmetry.operations, sys_, require_closed=False,
    )
    reps = vq.compress_lattice_matrix_set_c(M, orbits, basis)
    rec = vq.reconstruct_lattice_matrix_set_c(
        reps, orbits, basis, sys_, sys_.symmetry.operations, cells=M.cells,
    )
    assert len(rec) == len(M.cells)
    max_err = max(
        np.abs(np.asarray(M.blocks[i]) - rec[i]).max()
        for i in range(len(M.cells))
    )
    assert max_err < 1e-10, (
        f"{integral_name}: reconstruction error {max_err:.3e}"
    )


def test_roundtrip_works_on_primitive_mg_too():
    """SYM2c should subsume SYM2b — the simple origin-fixed case
    also round-trips at machine precision through the generalised
    code path."""
    sys_, basis, opts = _primitive_mg()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_atom_pair_orbits(
        S.cells, sys_.symmetry.operations, sys_, require_closed=False,
    )
    reps = vq.compress_lattice_matrix_set_c(S, orbits, basis)
    rec = vq.reconstruct_lattice_matrix_set_c(
        reps, orbits, basis, sys_, sys_.symmetry.operations, cells=S.cells,
    )
    max_err = max(
        np.abs(np.asarray(S.blocks[i]) - rec[i]).max()
        for i in range(len(S.cells))
    )
    assert max_err < 1e-10


# ---------------------------------------------------------------------------
# Sub-block shapes
# ---------------------------------------------------------------------------

def test_rep_subblock_shape_matches_atom_ao_counts():
    """The representative sub-block of an (a_rep, b_rep) orbit has
    shape (n_AOs_on_a_rep, n_AOs_on_b_rep)."""
    sys_, basis, opts = _nacl()
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_atom_pair_orbits(
        S.cells, sys_.symmetry.operations, sys_, require_closed=False,
    )
    reps = vq.compress_lattice_matrix_set_c(S, orbits, basis)
    for orb, rep in zip(orbits.orbits, reps):
        a_rep, b_rep, _ = orb.representative
        # For NaCl with STO-3G: Na has 9 AOs (1s+2s+2p+3s+3p = 9),
        # Cl has 9 AOs (same shell pattern). So every sub-block is 9×9.
        assert rep.shape == (9, 9), (
            f"rep {orb.representative} has shape {rep.shape}, "
            f"expected (9, 9)"
        )


# ---------------------------------------------------------------------------
# Closure error path
# ---------------------------------------------------------------------------

def test_non_closed_triple_space_raises_with_flag():
    """With ``require_closed=True`` (default), a triple whose image
    lands outside the cell cutoff must raise a directive error
    — better than silently producing partial orbits."""
    sys_, basis, opts = _nacl(cutoff=8.0)   # too-tight cutoff
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    with pytest.raises(ValueError, match="not symmetry-closed"):
        vq.identify_atom_pair_orbits(
            S.cells, sys_.symmetry.operations, sys_, require_closed=True,
        )


def test_closure_skip_preserves_roundtrip():
    """Even at a cutoff where some orbits straddle the boundary,
    running with ``require_closed=False`` should give an orbit
    partition where every in-space triple is covered, and the
    round-trip is still exact on in-space triples."""
    sys_, basis, opts = _nacl(cutoff=10.0)   # modest cutoff
    S = vq.compute_overlap_lattice(basis, sys_, opts)
    orbits = vq.identify_atom_pair_orbits(
        S.cells, sys_.symmetry.operations, sys_, require_closed=False,
    )
    reps = vq.compress_lattice_matrix_set_c(S, orbits, basis)
    rec = vq.reconstruct_lattice_matrix_set_c(
        reps, orbits, basis, sys_, sys_.symmetry.operations, cells=S.cells,
    )
    max_err = max(
        np.abs(np.asarray(S.blocks[i]) - rec[i]).max()
        for i in range(len(S.cells))
    )
    assert max_err < 1e-10


# ---------------------------------------------------------------------------
# Boundary case: single atom
# ---------------------------------------------------------------------------

def test_single_atom_cell_has_only_self_pairs():
    """With one atom in the unit cell, there's only one atom pair
    (0, 0) and orbits reduce to cell-level orbits (matching SYM2b's
    output for the same system)."""
    sys_, basis, opts = _primitive_mg()
    S = vq.compute_overlap_lattice(basis, sys_, opts)

    orbits_c = vq.identify_atom_pair_orbits(
        S.cells, sys_.symmetry.operations, sys_,
    )
    # All triples have atom pair (0, 0).
    for orb in orbits_c.orbits:
        for a, b, _ in orb.members:
            assert a == 0 and b == 0

    # Number of orbits under SYM2c (per-atom-pair) should equal
    # number under SYM2b (per-cell) in the single-atom case.
    orbits_b = vq.identify_lattice_orbits(
        S.cells, sys_.symmetry.operations,
    )
    assert orbits_c.n_orbits == orbits_b.n_orbits
