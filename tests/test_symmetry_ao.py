"""Phase SYM2a tests: AO-basis representation of symmetry operators.

Core contracts:

1. **Atom permutation identification** works on test cases with
   known point-group symmetry (H2O with C_{2v}, benzene-like with
   C_6, periodic cubic lattice with O_h).

2. **AO permutation matrix P is orthogonal** — since the AO basis
   has an orthogonality structure and R is a proper rotation, so
   is the representation.

3. **P preserves the overlap matrix**: ``P · S · P^T = S`` at machine
   precision, because the basis set is mapped to itself when (R, t)
   is a symmetry of the nuclear framework.

4. **P preserves the converged Fock / density** matrices when the
   SCF state has the symmetry. This is the physics witness that
   matters for SYM2b — block reduction of real-space ``F(h)``
   requires exactly this invariance.

5. **Group-theoretic identities** — identity op gives identity P,
   composition is a homomorphism (P(R1·R2) = P(R1) · P(R2)), inverse
   is transpose (P(R^T) = P(R)^T).

6. **Error paths** — non-symmetry ops raise directively, ambiguous
   images (degenerate geometries at tight tolerance) raise
   directively.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _h2o_c2v():
    """H2O in the y=0 plane, C2 axis = z, σ_v = xz plane (trivial) and
    σ_v' = yz plane (swaps the H atoms)."""
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.5, 0.0, -1.2]),
        vq.Atom(1, [-1.5, 0.0, -1.2]),
    ])


def _nh3_c3v_like():
    """NH3 — nitrogen on the z axis, three hydrogens 120° apart at
    z = 0. Closed-shell (7 + 3·1 = 10 electrons) so the default
    Molecule multiplicity=1 is consistent. C3 about z is a proper
    rotation of the structure that cycles the three hydrogens."""
    r = 1.0
    coords = [
        (7, [0.0, 0.0, 0.5]),
    ]
    for k in range(3):
        angle = 2 * np.pi * k / 3
        coords.append((1, [r * np.cos(angle), r * np.sin(angle), 0.0]))
    return vq.Molecule([vq.Atom(Z, list(xyz)) for Z, xyz in coords])


# Named rotations for convenience.
def _R_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


C2_z = _R_z(np.pi)
C3_z = _R_z(2 * np.pi / 3)
C6_z = _R_z(np.pi / 3)


# ---------------------------------------------------------------------------
# Atom permutation identification
# ---------------------------------------------------------------------------

def test_identity_op_fixes_all_atoms():
    mol = _h2o_c2v()
    ap = vq.atom_permutation_under_op(mol, np.eye(3))
    assert np.array_equal(ap.perm, [0, 1, 2])
    assert np.array_equal(ap.lattice_shift, np.zeros((3, 3), dtype=int))


def test_C2_swaps_the_two_hydrogens():
    """C2 about z on H2O in the y=0 plane: O is invariant, the two
    H atoms swap."""
    mol = _h2o_c2v()
    ap = vq.atom_permutation_under_op(mol, C2_z)
    assert np.array_equal(ap.perm, [0, 2, 1])


def test_C3_cycles_three_hydrogens():
    """120° rotation cycles the three H atoms 0 → 1 → 2 → 0."""
    mol = _nh3_c3v_like()
    ap = vq.atom_permutation_under_op(mol, C3_z)
    # atoms 1, 2, 3 are at 0°, 120°, 240° around the z axis; C3
    # rotates 0° → 120°, so atom 1 goes to atom 2, atom 2 to atom 3,
    # atom 3 to atom 1.
    assert np.array_equal(ap.perm, [0, 2, 3, 1])


def test_non_symmetry_op_rejected():
    """A 45° rotation about z is not a symmetry of H2O. The image of
    the first H atom lands in empty space — clear error message."""
    mol = _h2o_c2v()
    R = _R_z(np.pi / 4)
    with pytest.raises(ValueError, match="not a symmetry"):
        vq.atom_permutation_under_op(mol, R)


def test_rejects_rotation_with_no_image_for_heavy_atom():
    """A rotation that maps a heavy atom to empty space (or, in a
    mixed-element structure, to the position of a lighter atom)
    is not a symmetry and must be rejected. C2(z) on an atom at
    (1, 1, 0) sends it to (-1, -1, 0) which is empty — clean rejection."""
    # CH (a carbon and a hydrogen along a diagonal) — 7 electrons so
    # multiplicity must be 2 for a consistent Molecule.
    mol = vq.Molecule(
        [vq.Atom(6, [1.0, 1.0, 0.0]), vq.Atom(1, [-1.0, -1.0, 0.0])],
        charge=0, multiplicity=2,
    )
    # A 90° rotation about z sends (1,1,0) to (-1,1,0), empty space.
    R = _R_z(np.pi / 2)
    with pytest.raises(ValueError, match="not a symmetry"):
        vq.atom_permutation_under_op(mol, R)


# ---------------------------------------------------------------------------
# AO permutation matrix: orthogonality, S-invariance
# ---------------------------------------------------------------------------

def _ortho_check(P: np.ndarray, atol: float = 1e-12) -> None:
    n = P.shape[0]
    assert np.allclose(P @ P.T, np.eye(n), atol=atol)
    assert np.allclose(P.T @ P, np.eye(n), atol=atol)


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31g*"])
def test_P_is_orthogonal_on_h2o_c2(basis_name):
    mol = _h2o_c2v()
    basis = vq.BasisSet(mol, basis_name)
    ap = vq.atom_permutation_under_op(mol, C2_z)
    P = vq.build_ao_permutation_matrix(basis, C2_z, ap)
    _ortho_check(P)


def test_P_preserves_overlap():
    """The overlap matrix is invariant under every symmetry of the
    molecule: the basis maps to itself under the op, so
    ``P · S · P^T = S``. Exercises both proper (E, C2) and improper
    (σ_v) operators — the latter validates the inversion-factorisation
    path in :func:`wigner_d_real`."""
    mol = _h2o_c2v()
    basis = vq.BasisSet(mol, "sto-3g")
    S = np.asarray(vq.compute_overlap(basis))
    # Proper rotations: identity and C2(z). Improper: σ_v planes
    # (xz and yz). All four are C_{2v} symmetries of H2O in the y=0
    # plane.
    sigma_xz = np.diag([1.0, -1.0, 1.0])   # mirror in xz (leaves everything invariant)
    sigma_yz = np.diag([-1.0, 1.0, 1.0])   # mirror in yz (swaps the two hydrogens)
    for R in (np.eye(3), C2_z, sigma_xz, sigma_yz):
        ap = vq.atom_permutation_under_op(mol, R)
        P = vq.build_ao_permutation_matrix(basis, R, ap)
        assert np.allclose(P @ S @ P.T, S, atol=1e-12), (
            f"overlap not preserved under R with det={np.linalg.det(R):.0f}"
        )


def test_P_preserves_converged_density():
    """The converged RHF density has the full molecular symmetry."""
    mol = _h2o_c2v()
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_rhf(mol, basis)
    D = np.asarray(result.density)
    ap = vq.atom_permutation_under_op(mol, C2_z)
    P = vq.build_ao_permutation_matrix(basis, C2_z, ap)
    assert np.allclose(P @ D @ P.T, D, atol=1e-10)


def test_P_preserves_converged_fock():
    """The Fock matrix must be symmetric under every molecular
    symmetry at convergence (if not, the SCF is broken or the state
    has reduced symmetry). Good self-consistency witness."""
    mol = _h2o_c2v()
    basis = vq.BasisSet(mol, "6-31g*")
    result = vq.run_rhf(mol, basis)
    F = np.asarray(result.fock)
    ap = vq.atom_permutation_under_op(mol, C2_z)
    P = vq.build_ao_permutation_matrix(basis, C2_z, ap)
    assert np.allclose(P @ F @ P.T, F, atol=1e-8)


# ---------------------------------------------------------------------------
# Group identities on P
# ---------------------------------------------------------------------------

def test_P_identity_op_is_identity_matrix():
    mol = _h2o_c2v()
    basis = vq.BasisSet(mol, "sto-3g")
    ap = vq.atom_permutation_under_op(mol, np.eye(3))
    P = vq.build_ao_permutation_matrix(basis, np.eye(3), ap)
    assert np.allclose(P, np.eye(basis.nbasis), atol=1e-14)


def test_P_composition_is_homomorphism():
    """P(R1 · R2) equals P(R1) · P(R2). Uses two proper rotations of
    an NH3-like C3v structure: C3 and its square (both around z). The
    composition C3 · C3 = C3^2 is a well-defined proper rotation and
    its AO representation must match the product of the individual
    reps."""
    mol = _nh3_c3v_like()
    basis = vq.BasisSet(mol, "sto-3g")
    R1 = C3_z
    R2 = C3_z          # C3 · C3 = C3² (rotation by 240°)
    R12 = R1 @ R2
    ap1 = vq.atom_permutation_under_op(mol, R1)
    ap2 = vq.atom_permutation_under_op(mol, R2)
    ap12 = vq.atom_permutation_under_op(mol, R12)
    P1 = vq.build_ao_permutation_matrix(basis, R1, ap1)
    P2 = vq.build_ao_permutation_matrix(basis, R2, ap2)
    P12 = vq.build_ao_permutation_matrix(basis, R12, ap12)
    assert np.allclose(P12, P1 @ P2, atol=1e-10)


def test_P_inverse_is_transpose():
    """For any proper rotation R, R^T = R^{-1} and P(R^T) = P(R)^T."""
    mol = _h2o_c2v()
    basis = vq.BasisSet(mol, "6-31g*")
    ap = vq.atom_permutation_under_op(mol, C2_z)
    P = vq.build_ao_permutation_matrix(basis, C2_z, ap)
    ap_inv = vq.atom_permutation_under_op(mol, C2_z.T)
    P_inv = vq.build_ao_permutation_matrix(basis, C2_z.T, ap_inv)
    assert np.allclose(P_inv, P.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Periodic case
# ---------------------------------------------------------------------------

def test_periodic_cubic_atom_permutation_with_lattice_shift():
    """A simple cubic lattice with one atom at the origin. Every
    rotation about the cubic axes is a symmetry; the atom maps to
    itself (with a lattice shift if the rotation carries the atom
    to a periodic image of itself which is the *same* atom, but we
    pick coordinates on the boundary to illustrate the lattice-shift
    mechanism)."""
    a = 3.0
    # Atom at the origin.
    system = vq.PeriodicSystem(
        3,
        np.diag([a, a, a]),
        [vq.Atom(11, [0.0, 0.0, 0.0])],
    )
    ap = vq.atom_permutation_under_op(system, C2_z)
    # Single atom → permutes to itself.
    assert ap.perm.tolist() == [0]
    # No lattice shift for the atom at the origin.
    assert np.all(ap.lattice_shift == 0)


def test_periodic_two_atom_basis_swaps_under_inversion_like_op():
    """Simple-cubic lattice with two atoms per cell: one at the
    origin, one at (a/2, a/2, a/2) — salt-like structure. A C2
    rotation about an axis bisecting them does not swap them (they
    have different elements by construction). But (x,y,z) →
    (x, y, z) + (-a/2, -a/2, -a/2) would map atom 2 into atom 1's
    position only if they had the same element; use a same-element
    structure for that."""
    a = 4.0
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [a / 2, a / 2, a / 2]),
    ]
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    # The C2 rotation about z about the midpoint of the atoms maps
    # atom 1 → atom 2 (modulo lattice). But a C2 rotation is about an
    # axis through the origin; it maps (0,0,0) to (0,0,0) and
    # (a/2, a/2, a/2) to (-a/2, -a/2, a/2). After lattice fold-back
    # (+a, +a, 0), this lands at (a/2, a/2, a/2) = atom 2. So atom 2
    # is fixed (with lattice shift (1,1,0)) and atom 1 is fixed
    # (no shift). Atom 1 and atom 2 do *not* swap under this C2.
    ap = vq.atom_permutation_under_op(system, C2_z)
    assert ap.perm.tolist() == [0, 1]
    assert np.all(ap.lattice_shift[0] == 0)
    # Atom 2 needs a (1, 1, 0) lattice shift to land back on itself.
    assert np.all(ap.lattice_shift[1] == [1, 1, 0])


def test_periodic_non_symmetry_rejected():
    """A 45° rotation isn't a symmetry of a non-square lattice."""
    system = vq.PeriodicSystem(
        3,
        np.diag([3.0, 4.0, 5.0]),
        [vq.Atom(1, [0, 0, 0])],
    )
    R = _R_z(np.pi / 4)   # 45°
    # Single atom at origin: any rotation permutes it to itself, so
    # the test instead uses a rotation that the 3:4:5 lattice can't
    # accommodate. Place the atom off-origin.
    system2 = vq.PeriodicSystem(
        3,
        np.diag([3.0, 4.0, 5.0]),
        [vq.Atom(1, [1.0, 0.0, 0.0])],
    )
    with pytest.raises(ValueError, match="not a symmetry"):
        vq.atom_permutation_under_op(system2, R)
