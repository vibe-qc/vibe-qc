"""Two-center CCM integrals (overlap, kinetic) — AICCM milestone 1.

Validates the CCM-weighted ``S^CCM`` / ``T^CCM`` built by folding vibe-qc's
real-space lattice blocks against the Wigner–Seitz weights:

* structural properties (symmetry, unit diagonal, circulant for a ring);
* the fast, translational-symmetry-exploiting builder reproduces a simple
  brute-force oracle for **all crystal lattices**;
* the physical Γ-point property: the eigenvalues of ``S^CCM`` converge to
  the infinite-crystal Bloch overlap ``S(k_j)`` at the folded k-points as
  the cluster grows (Peintinger & Bredow 2014, Tables 5–6);
* the overlap-spectrum guard (Fig. 6).

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550, eqs (5)–(6).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    LatticeSumOptions,
    PeriodicSystem,
    bloch_sum,
    compute_overlap_lattice,
)
from vibeqc._vibeqc_core import Molecule
from vibeqc.periodic.ccm import CCMSystem, ccm_kinetic, ccm_overlap
from vibeqc.periodic.ccm.wigner_seitz import wigner_seitz_weights_reference

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


def _periodic_cell(lattice, atoms):
    mult = 1 if sum(a.Z for a in atoms) % 2 == 0 else 2
    return PeriodicSystem(3, np.asarray(lattice, float), atoms, charge=0, multiplicity=mult)


def _reference_overlap(ccm):
    """Brute-force CCM overlap oracle (independent of the fast builder)."""
    lms = compute_overlap_lattice(ccm.basis, ccm.cluster_system, ccm.lattice_options)
    R = np.array([list(c.r_cart) for c in lms.cells])
    w = wigner_seitz_weights_reference(ccm.atom_positions, R, tol_bohr=ccm.weight_tol_bohr)
    ao = ccm.ao_atom
    out = np.zeros((ccm.nbf, ccm.nbf))
    for g, blk in enumerate(lms.blocks):
        out += w[g][ao[:, None], ao[None, :]] * np.asarray(blk, float)
    return 0.5 * (out + out.T)


def _h_chain(spacing_bohr, n_cells):
    unit = _periodic_cell(np.diag([spacing_bohr, 30.0, 30.0]), [Atom(1, [0, 0, 0])])
    return CCMSystem(unit, (n_cells, 1, 1), "sto-3g")


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def test_overlap_symmetric_unit_diagonal():
    ccm = _h_chain(1.0 * BOHR, 6)
    S = ccm_overlap(ccm)
    assert S.shape == (ccm.nbf, ccm.nbf)
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0, atol=1e-8)  # normalised STO-3G s functions


def test_kinetic_symmetric():
    ccm = _h_chain(1.0 * BOHR, 6)
    T = ccm_kinetic(ccm)
    assert np.allclose(T, T.T)
    assert np.all(np.diag(T) > 0.0)


def test_overlap_circulant_for_ring():
    """Equidistant H ring → S^CCM is circulant (every atom equivalent)."""
    n = 6
    ccm = _h_chain(1.0 * BOHR, n)
    S = ccm_overlap(ccm)
    for i in range(n):
        for j in range(n):
            assert S[i, j] == pytest.approx(S[0, (j - i) % n], abs=1e-9)


# --------------------------------------------------------------------------- #
# Fast builder == brute-force oracle, for all crystal lattices
# --------------------------------------------------------------------------- #
ah, ch = 4.5, 6.0
CASES = {
    "cubic": (np.diag([3.0, 3.0, 3.0]), [Atom(1, [0, 0, 0])], (3, 3, 3)),
    "chain": (np.diag([2.0, 30.0, 30.0]), [Atom(1, [0, 0, 0])], (6, 1, 1)),
    "hexagonal": (
        [[ah, 0, 0], [-ah / 2, ah * np.sqrt(3) / 2, 0], [0, 0, ch]],
        [Atom(1, [0, 0, 0]), Atom(1, [ah * 0.5, ah * 0.2, ch * 0.5])],
        (3, 3, 2),
    ),
    "triclinic": (
        [[4.0, 0, 0], [1.6, 3.7, 0], [1.2, 0.9, 4.3]],
        [Atom(1, [0, 0, 0]), Atom(1, [1.0, 0.5, 1.5])],
        (3, 3, 3),
    ),
    "fcc_primitive": (
        [[0, 2.5, 2.5], [2.5, 0, 2.5], [2.5, 2.5, 0]],
        [Atom(2, [0, 0, 0])],
        (3, 3, 3),
    ),
    "monoclinic": (
        [[5.0, 0, 0], [0, 4.0, 0], [1.5, 0, 6.0]],
        [Atom(1, [0, 0, 0]), Atom(1, [2.0, 2.0, 3.0])],
        (2, 3, 2),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_fast_builder_matches_oracle(name):
    lattice, atoms, nrep = CASES[name]
    ccm = CCMSystem(_periodic_cell(lattice, atoms), nrep, "sto-3g")
    # Atom (cell, beta) decomposition must reconstruct the supercell positions.
    recon = ccm.unit_atom_pos[ccm.atom_beta] + ccm.atom_cell @ ccm.unit_vectors
    assert np.allclose(recon, ccm.atom_positions, atol=1e-9)
    # Fast symmetry-exploiting fold == brute-force oracle.
    assert np.max(np.abs(ccm_overlap(ccm) - _reference_overlap(ccm))) < 1e-10


# --------------------------------------------------------------------------- #
# Physical Γ-point property: convergence to the infinite-crystal Bloch overlap
# --------------------------------------------------------------------------- #
def _bloch_overlap_eigs(spacing_bohr, n_cells):
    unit = _periodic_cell(np.diag([spacing_bohr, 30.0, 30.0]), [Atom(1, [0, 0, 0])])
    basis = BasisSet(Molecule([Atom(1, [0, 0, 0])], 0, 2), "sto-3g")
    opts = LatticeSumOptions()
    opts.cutoff_bohr = max(60.0, 3 * n_cells * spacing_bohr)
    lms = compute_overlap_lattice(basis, unit, opts)
    vals = []
    for j in range(n_cells):
        kx = 2.0 * np.pi * j / (n_cells * spacing_bohr)
        sk = np.asarray(bloch_sum(lms, np.array([kx, 0.0, 0.0])))
        vals.append(float(np.real(sk[0, 0])))
    return np.sort(vals)


def test_overlap_eigs_converge_to_bloch():
    """eig(S^CCM) → infinite-chain Bloch S(k_j) as the cluster grows."""
    a = 1.0 * BOHR
    errs = []
    for n in (6, 10, 16):
        eig = np.sort(np.linalg.eigvalsh(ccm_overlap(_h_chain(a, n))))
        errs.append(np.max(np.abs(eig - _bloch_overlap_eigs(a, n))))
    # Monotone convergence, essentially exact once the cluster exceeds the
    # overlap range (half-period overlap -> 0).
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 1e-8


# --------------------------------------------------------------------------- #
# Overlap-spectrum guard (Fig. 6)
# --------------------------------------------------------------------------- #
def test_overlap_spectrum_guard():
    ccm = _h_chain(1.0 * BOHR, 6)
    emin, evals = ccm.check_overlap_spectrum(threshold=1e-10)
    assert emin > 0.0
    assert evals[0] == pytest.approx(emin)
    # A threshold above the true minimum eigenvalue must trip the guard.
    with pytest.raises(ValueError, match="near-singular"):
        ccm.check_overlap_spectrum(threshold=10.0)
