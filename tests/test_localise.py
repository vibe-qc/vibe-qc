"""Physical-correctness tests for orbital localisation (Foster-Boys, Pipek-Mezey).

Beyond unitarity, these tests assert the *physics* of localisation on
H2O/STO-3G: the five localised occupied orbitals must form the textbook
pattern — one O core, two equivalent O-H bond orbitals, and two
equivalent lone pairs — and the localisation objective must strictly
increase relative to the canonical orbitals.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, compute_dipole, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule, compute_overlap
from vibeqc.localise import (
    boys_objective,
    foster_boys_localise,
    pipek_mezey_localise,
    pipek_mezey_objective,
)

ANGSTROM_TO_BOHR = 1.8897259886

# H2O, positions in bohr (same convention as tests/conftest.py GEOMETRIES):
# O at origin, hydrogens at (0, ±y, -z) — bonds point "down" in z,
# lone pairs point "up".
H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]


def _build_atom_basis_map(basis, natom: int) -> np.ndarray:
    """Boolean (nbf, natom) map: AO μ belongs to atom A (spherical shells)."""
    nbf = basis.nbasis
    amap = np.zeros((nbf, natom))
    bf = 0
    for sh in basis.shells():
        n_func = 2 * int(sh.l) + 1
        amap[bf : bf + n_func, int(sh.atom_index)] = 1.0
        bf += n_func
    return amap


@pytest.fixture(scope="module")
def h2o():
    """H2O/STO-3G RHF with real dipole integrals."""
    mol = Molecule(
        [Atom(z, pos) for z, pos in H2O_ATOMS], charge=0, multiplicity=1
    )
    basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged

    n_occ = mol.n_electrons() // 2
    C_occ = rhf.mo_coeffs[:, :n_occ].copy()

    dip = compute_dipole(basis)
    nbf = basis.nbasis
    dipoles = np.zeros((nbf, nbf, 3))
    dipoles[:, :, 0] = np.asarray(dip.x)
    dipoles[:, :, 1] = np.asarray(dip.y)
    dipoles[:, :, 2] = np.asarray(dip.z)

    S = np.asarray(compute_overlap(basis)).copy()
    return {
        "mol": mol,
        "basis": basis,
        "C_occ": C_occ,
        "dipoles": dipoles,
        "S": S,
        "n_occ": n_occ,
    }


def _centroids(C: np.ndarray, dipoles: np.ndarray) -> np.ndarray:
    """Orbital centroids ⟨i|r|i⟩ from exact dipole integrals."""
    n = C.shape[1]
    cen = np.zeros((n, 3))
    for c in range(3):
        cen[:, c] = np.einsum("mi,mn,ni->i", C, dipoles[:, :, c], C)
    return cen


class TestFosterBoys:
    def test_objective_strictly_increases(self, h2o):
        C, dipoles = h2o["C_occ"], h2o["dipoles"]
        C_loc = foster_boys_localise(C, dipoles)
        f0 = boys_objective(C, dipoles)
        f1 = boys_objective(C_loc, dipoles)
        # Canonical H2O orbitals are symmetry-delocalised; Boys must
        # gain a substantial margin, not a numerical whisker.
        assert f1 > f0 + 0.5

    def test_unitarity_and_span(self, h2o):
        C, dipoles, S = h2o["C_occ"], h2o["dipoles"], h2o["S"]
        C_loc = foster_boys_localise(C, dipoles)
        # Orthonormality preserved.
        np.testing.assert_allclose(
            C_loc.T @ S @ C_loc, np.eye(C.shape[1]), atol=1e-10
        )
        # Same occupied space: identical density matrix.
        np.testing.assert_allclose(C_loc @ C_loc.T, C @ C.T, atol=1e-10)

    def test_h2o_textbook_pattern(self, h2o):
        """1 core on O + 2 equivalent O-H bonds + 2 equivalent lone pairs."""
        C, dipoles = h2o["C_occ"], h2o["dipoles"]
        C_loc = foster_boys_localise(C, dipoles)
        cen = _centroids(C_loc, dipoles)

        r = np.linalg.norm(cen, axis=1)
        core = np.where(r < 0.25)[0]
        assert len(core) == 1, f"expected 1 core orbital at O, centroids:\n{cen}"

        rest = np.array([k for k in range(5) if k != core[0]])
        # Bond orbitals: displaced toward the hydrogens (y ≠ 0, z < 0).
        bonds = [k for k in rest if abs(cen[k, 1]) > 0.35 and cen[k, 2] < -0.15]
        # Lone pairs: out-of-plane rabbit ears (x ≠ 0, z > 0).
        lps = [k for k in rest if abs(cen[k, 0]) > 0.25 and cen[k, 2] > 0.05]
        assert len(bonds) == 2, f"expected 2 bond orbitals, centroids:\n{cen}"
        assert len(lps) == 2, f"expected 2 lone pairs, centroids:\n{cen}"

        # Equivalence by mirror symmetry: bonds mirror in y, LPs in x.
        b1, b2 = cen[bonds[0]], cen[bonds[1]]
        np.testing.assert_allclose(b1[1], -b2[1], atol=1e-3)
        np.testing.assert_allclose(b1[2], b2[2], atol=1e-3)
        l1, l2 = cen[lps[0]], cen[lps[1]]
        np.testing.assert_allclose(l1[0], -l2[0], atol=1e-3)
        np.testing.assert_allclose(l1[2], l2[2], atol=1e-3)

    def test_idempotent_on_localised_input(self, h2o):
        """Re-localising an already-localised set is a no-op (max angle ~0)."""
        C, dipoles = h2o["C_occ"], h2o["dipoles"]
        C_loc = foster_boys_localise(C, dipoles)
        C_loc2 = foster_boys_localise(C_loc, dipoles)
        f1 = boys_objective(C_loc, dipoles)
        f2 = boys_objective(C_loc2, dipoles)
        np.testing.assert_allclose(f2, f1, rtol=1e-9)


class TestPipekMezey:
    def test_objective_strictly_increases(self, h2o):
        C, S, basis, mol = h2o["C_occ"], h2o["S"], h2o["basis"], h2o["mol"]
        amap = _build_atom_basis_map(basis, len(list(mol.atoms)))
        C_loc = pipek_mezey_localise(C, S, amap)
        p0 = pipek_mezey_objective(C, S, amap)
        p1 = pipek_mezey_objective(C_loc, S, amap)
        assert p1 > p0 + 0.3

    def test_unitarity_and_span(self, h2o):
        C, S, basis, mol = h2o["C_occ"], h2o["S"], h2o["basis"], h2o["mol"]
        amap = _build_atom_basis_map(basis, len(list(mol.atoms)))
        C_loc = pipek_mezey_localise(C, S, amap)
        np.testing.assert_allclose(
            C_loc.T @ S @ C_loc, np.eye(C.shape[1]), atol=1e-10
        )
        np.testing.assert_allclose(C_loc @ C_loc.T, C @ C.T, atol=1e-10)

    def test_h2o_population_pattern(self, h2o):
        """PM keeps sigma/lone-pair separation: 3 orbitals ~pure O, 2 O-H bonds."""
        C, S, basis, mol = h2o["C_occ"], h2o["S"], h2o["basis"], h2o["mol"]
        natom = len(list(mol.atoms))
        amap = _build_atom_basis_map(basis, natom)
        C_loc = pipek_mezey_localise(C, S, amap)

        # Per-orbital Mulliken populations on each atom.
        CS = C_loc.T @ S
        Q = np.zeros((C.shape[1], natom))
        for a in range(natom):
            mask = amap[:, a] > 0.5
            Q[:, a] = np.einsum("im,mi->i", CS[:, mask], C_loc[mask, :])

        # Populations sum to 1 per orbital (normalised orbitals).
        np.testing.assert_allclose(Q.sum(axis=1), 1.0, atol=1e-8)

        on_oxygen = np.where(Q[:, 0] > 0.85)[0]
        assert len(on_oxygen) == 3, f"expected core+2 lone pairs on O, Q:\n{Q}"

        bonds = np.where(Q[:, 0] <= 0.85)[0]
        assert len(bonds) == 2
        # Each bond orbital lives on O plus exactly one hydrogen.
        h_share = Q[bonds][:, 1:]
        assert all(h_share[k].max() > 0.15 for k in range(2)), f"Q:\n{Q}"
        # The two bonds use different hydrogens.
        assert set(np.argmax(h_share, axis=1)) == {0, 1}
