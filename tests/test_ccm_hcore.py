"""CCM core Hamiltonian: three-center V^CCM + the padded-cluster path.

Milestone 2a. The padded-cluster molecular-integral path supplies the
home↔image and home↔point-charge integrals the lattice-block primitives
don't expose, and is the prototype the four-center ERIs will reuse. Tests:

* the padded two-center fold reproduces the verified lattice path (S, T) —
  anchoring the padded infrastructure against M1;
* V^CCM reduces to the ordinary molecular nuclear-attraction matrix for an
  isolated (huge-cell) cluster — an independent check of sign and of the
  union-WSSC limit;
* V^CCM matches an independent brute-force evaluation of eqs (12)-(13) for a
  periodic cluster;
* h^CCM = T^CCM + V^CCM assembles and is symmetric.

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550, eqs (12)-(13).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem, compute_nuclear
from vibeqc._vibeqc_core import Molecule
from vibeqc.periodic.ccm import CCMSystem, ccm_kinetic, ccm_overlap
from vibeqc.periodic.ccm.padded import (
    build_padded_cluster,
    ccm_hcore,
    ccm_kinetic_padded,
    ccm_nuclear,
    ccm_overlap_padded,
    wssc_cells,
)
from vibeqc.periodic.ccm.wigner_seitz import minimum_image

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _cell(lattice, atoms):
    mult = 1 if sum(a.Z for a in atoms) % 2 == 0 else 2
    return PeriodicSystem(3, np.asarray(lattice, float), atoms, charge=0, multiplicity=mult)


AH, CH = 4.5, 6.0
TWO_CENTER_CASES = {
    "chain": (np.diag([2.0, 30.0, 30.0]), [Atom(1, [0, 0, 0])], (6, 1, 1)),
    "hexagonal": ([[AH, 0, 0], [-AH / 2, AH * np.sqrt(3) / 2, 0], [0, 0, CH]],
                  [Atom(1, [0, 0, 0]), Atom(1, [AH * 0.5, AH * 0.2, CH * 0.5])], (2, 2, 2)),
    "triclinic": ([[4.0, 0, 0], [1.6, 3.7, 0], [1.2, 0.9, 4.3]],
                  [Atom(1, [0, 0, 0]), Atom(1, [1.0, 0.5, 1.5])], (2, 2, 2)),
}


@pytest.mark.parametrize("name", list(TWO_CENTER_CASES))
def test_padded_two_center_matches_lattice_fold(name):
    """Padded molecular fold == the verified lattice fold for S and T."""
    lattice, atoms, nrep = TWO_CENTER_CASES[name]
    ccm = CCMSystem(_cell(lattice, atoms), nrep, "sto-3g")
    assert np.max(np.abs(ccm_overlap_padded(ccm) - ccm_overlap(ccm))) < 1e-12
    assert np.max(np.abs(ccm_kinetic_padded(ccm) - ccm_kinetic(ccm))) < 1e-12


def test_isolated_cluster_V_equals_molecular():
    """Huge isolated cell: V^CCM is the ordinary molecular nuclear attraction."""
    unit = _cell(np.diag([60.0, 60.0, 60.0]),
                 [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    ccm = CCMSystem(unit, (1, 1, 1), "sto-3g")
    v_ccm = ccm_nuclear(ccm)
    v_mol = compute_nuclear(ccm.basis, ccm.supercell)
    assert np.allclose(v_ccm, v_mol, atol=1e-10)
    assert np.all(np.diag(v_ccm) < 0.0)  # attractive


def _v_oracle(ccm, pad):
    """Independent brute-force evaluation of V^CCM (eqs 12-13)."""
    pos, ao, nbf = ccm.atom_positions, ccm.ao_atom, ccm.nbf
    Ac, tol = ccm.cluster_vectors, ccm.weight_tol_bohr
    cache = {}

    def integral(ci):
        if ci not in cache:
            z = int(pad.pad_Z[ci])
            cache[ci] = compute_nuclear(
                pad.basis, Molecule([Atom(z, pad.pad_positions[ci].tolist())], 0,
                                    1 if z % 2 == 0 else 2))
        return cache[ci]

    def minimg(a, b):
        cells, w = minimum_image(pos[a] - pos[b], Ac, tol_bohr=tol)
        return [tuple(int(x) for x in c) for c in cells[0]], list(w[0])

    V = np.zeros((nbf, nbf))
    for mu in range(nbf):
        A = int(ao[mu])
        for nu in range(nbf):
            B = int(ao[nu])
            nu_cells, nu_w = minimg(A, B)
            acc = 0.0
            for g_nu, w_mn in zip(nu_cells, nu_w):
                col = pad.cell_to_cols[g_nu][nu]
                for ci, (C0, gc) in enumerate(pad.pad_atom_of):
                    cA, wA = minimg(A, C0)
                    cB, wB = minimg(B, C0)
                    oA = wA[cA.index(gc)] if gc in cA else 0.0
                    oB = wB[cB.index(gc)] if gc in cB else 0.0
                    if oA == 0.0 and oB == 0.0:
                        continue
                    acc += w_mn * 0.5 * (oA + oB) * integral(ci)[mu, col]
            V[mu, nu] = acc
    return 0.5 * (V + V.T)


def test_V_matches_oracle_periodic():
    ccm = CCMSystem(_cell(np.diag([2.2, 30.0, 30.0]), [Atom(1, [0, 0, 0])]), (4, 1, 1), "sto-3g")
    pad = build_padded_cluster(ccm, wssc_cells(ccm))
    assert np.max(np.abs(ccm_nuclear(ccm) - _v_oracle(ccm, pad))) < 1e-10


def test_hcore_assembles_symmetric():
    ccm = CCMSystem(_cell(np.diag([2.2, 30.0, 30.0]), [Atom(1, [0, 0, 0])]), (4, 1, 1), "sto-3g")
    h, t, v = ccm_hcore(ccm)
    assert np.allclose(h, t + v)
    assert np.allclose(h, h.T)
