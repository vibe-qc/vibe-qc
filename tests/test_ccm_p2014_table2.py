"""The faithful 2012/2014 AICCM selector, pinned against P2014 Table 2.

``method="union12"`` keeps the bra partner of the four-centre term in the home
copy of the cluster. P2014 Eq. 19 and the original program
(``~/gitlab/aiccm``, ``integrals/eri_functions.pyx::calcERIsUnion12``) sum it
over the closed Wigner-Seitz supercell of the first function, images included,
with the bridge weight relative to that imaged partner. ``method="p2014"``
(``padded.ccm_eri_p2014``, C++ kernel ``"p2014-direct"``) is that construction,
with the 2012 program's nuclear attraction (``"program-union"``: nucleus weight
relative to the imaged partner, no pair weight) as its default and the printed
Eq. 13 (``"printed-eq13"``) as the named alternative.

Reference: Peintinger and Bredow, J. Comput. Chem. 2014, Table 2 (alternating
hydrogen chain, four atoms at 0, 0.8, 2.0, 2.8 Angstrom, period 4.0 Angstrom,
STO-3G, hartree per atom), and the paper chat's numpy replica of the 2012
program (paper repository, ``theory/analysis-2026-09-17/aiccm2014.py``,
recorded 2026-09-17), which was computed with 1 Angstrom = 1/0.52917721092
bohr; this file uses the same constant so the replica column is comparable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm import padded as P
from vibeqc.periodic.ccm.scf import run_ccm_rhf, run_ccm_rhf_scalable

pytestmark = [pytest.mark.experimental, pytest.mark.filterwarnings("ignore")]  # 2014-lineage diagnosis

ANG = 1.0 / 0.52917721092

#: cells -> (P2014 Table 2, replica of the 2012 program), hartree per atom.
TABLE2 = {
    2: (-0.540663, -0.5406630457),
    4: (-0.542819, -0.5428186389),
    6: (-0.542873, -0.5428725423),
    8: (-0.542875, -0.5428747554),
    10: (-0.542875, -0.5428749521),
}
#: Printed Eq. 13 (with the pair weight) at two cells, replica value.
PRINTED_EQ13_2CELLS = -0.5403608
#: Structure numbers of the raw tensor recorded by the replica: cells ->
#: (ket asymmetry, bra-ket asymmetry), relative Frobenius norms.
RAW_ASYMMETRY = {2: (1.424e-2, 1.007e-2), 4: (3.269e-3, 2.312e-3)}


def _chain(cells):
    lattice = np.diag([4.0 * ANG, 40.0, 40.0])
    atoms = [Atom(1, [x * ANG, 0.0, 0.0]) for x in (0.0, 0.8, 2.0, 2.8)]
    unit = PeriodicSystem(1, lattice, atoms, 0, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CCMSystem(unit, (cells, 1, 1), "sto-3g")


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


@pytest.mark.parametrize("cells", [2, 4, 6])
def test_dense_p2014_reproduces_table2(cells):
    published, replica = TABLE2[cells]
    r = run_ccm_rhf(_chain(cells), method="p2014", conv_tol=1e-12,
                    conv_tol_grad=1e-9)
    assert r.converged
    assert r.backend == "ccm-fourcenter-dense-p2014-rhf"
    assert r.nuclear_attraction == "program-union"
    print(f"cells={cells} E/atom={r.energy_per_atom!r} "
          f"vs replica {r.energy_per_atom - replica:+.2e}")
    assert abs(r.energy_per_atom - published) < 1e-6
    assert abs(r.energy_per_atom - replica) < 1e-8


@pytest.mark.parametrize("cells", [2, 4, 6, 8, 10])
def test_direct_kernel_p2014_reproduces_table2(cells):
    published, replica = TABLE2[cells]
    r = run_ccm_rhf_scalable(_chain(cells), method="p2014", conv_tol=1e-12,
                             conv_tol_grad=1e-9)
    assert r.converged
    assert r.backend == "ccm-fourcenter-direct-p2014-rhf"
    assert r.nuclear_attraction == "program-union"
    print(f"cells={cells} E/atom={r.energy_per_atom!r} "
          f"vs replica {r.energy_per_atom - replica:+.2e}")
    assert abs(r.energy_per_atom - published) < 1e-6
    assert abs(r.energy_per_atom - replica) < 1e-8


@pytest.mark.parametrize("cells", [2, 4])
def test_direct_kernel_equals_dense_tensor(cells):
    ccm = _chain(cells)
    d = run_ccm_rhf(ccm, method="p2014", conv_tol=1e-12, conv_tol_grad=1e-9)
    s = run_ccm_rhf_scalable(ccm, method="p2014", conv_tol=1e-12,
                             conv_tol_grad=1e-9)
    assert abs(d.energy - s.energy) < 1e-10


@pytest.mark.parametrize("cells", [2, 4])
def test_structure_of_the_p2014_tensor(cells):
    """First-pair symmetry of V, eight-fold symmetry of (V_ijkl + V_ijlk)/2,
    and a self-adjoint assembled two-electron map on symmetric densities."""
    ccm = _chain(cells)
    V = P.ccm_eri_p2014(ccm, ket_symmetrize=False)
    assert _rel(V, V.transpose(1, 0, 2, 3)) < 1e-12
    ket, braket = RAW_ASYMMETRY[cells]
    assert _rel(V, V.transpose(0, 1, 3, 2)) == pytest.approx(ket, rel=2e-3)
    assert _rel(V, V.transpose(2, 3, 0, 1)) == pytest.approx(braket, rel=2e-3)

    Vb = P.ccm_eri_p2014(ccm)
    assert np.array_equal(Vb, 0.5 * (V + V.transpose(0, 1, 3, 2)))
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        assert _rel(Vb, Vb.transpose(perm)) < 1e-12

    # G: D_a -> 2 J(D_a) - K(D_a), restricted to Sym(d) -> Sym(d).
    g = 2.0 * Vb - Vb.transpose(0, 2, 1, 3)
    g = 0.5 * (g + g.transpose(0, 1, 3, 2))
    g = 0.5 * (g + g.transpose(1, 0, 2, 3))
    assert _rel(g, g.transpose(2, 3, 0, 1)) < 1e-12


def test_nuclear_attraction_variants_are_named_and_distinct():
    ccm2 = _chain(2)
    printed = run_ccm_rhf(ccm2, method="p2014", nuclear_attraction="printed-eq13",
                          conv_tol=1e-12, conv_tol_grad=1e-9)
    assert printed.nuclear_attraction == "printed-eq13"
    assert abs(printed.energy_per_atom - PRINTED_EQ13_2CELLS) < 5e-7
    # ... and the published two-cell value is the program's, not the printed one.
    assert abs(printed.energy_per_atom - TABLE2[2][0]) > 2.5e-4

    ccm4 = _chain(4)
    a = run_ccm_rhf(ccm4, method="p2014", conv_tol=1e-12, conv_tol_grad=1e-9)
    b = run_ccm_rhf(ccm4, method="p2014", nuclear_attraction="printed-eq13",
                    conv_tol=1e-12, conv_tol_grad=1e-9)
    assert abs(a.energy_per_atom - b.energy_per_atom) < 1e-8

    for variant in P.P2014_NUCLEAR_VARIANTS:
        raw = P.ccm_nuclear_p2014(ccm2, variant=variant, symmetrize=False)
        assert np.abs(raw - raw.T).max() < 1e-12


def test_union12_stays_reachable_and_is_not_the_2012_algorithm():
    """Archived records depend on "union12"; it misses the published ladder
    (1.9e-4 and 1.4e-4 hartree per atom at two and four cells)."""
    for cells, miss in ((2, 1.93e-4), (4, 1.43e-4)):
        r = run_ccm_rhf(_chain(cells), method="union12", conv_tol=1e-12,
                        conv_tol_grad=1e-9)
        assert r.nuclear_attraction == "eq13-home-image"
        assert r.energy_per_atom - TABLE2[cells][1] == pytest.approx(miss, rel=2e-2)


def test_p2014_is_refused_where_it_would_be_mixed_with_the_old_nuclear_term():
    from vibeqc.periodic.ccm.scf import (
        _ccm_eri_for_method, _ccm_scalable_cxx_method)

    ccm = _chain(2)
    with pytest.raises(ValueError, match="p2014"):
        _ccm_eri_for_method(ccm, "p2014")        # MP2/CCSD/gradient registry
    with pytest.raises(ValueError, match="run_ccm_rhf"):
        _ccm_scalable_cxx_method("p2014", "direct")   # the KS builder's path
    with pytest.raises(ValueError, match="integral-direct"):
        run_ccm_rhf_scalable(ccm, method="p2014", four_center="full")
    with pytest.raises(ValueError, match="only for"):
        run_ccm_rhf(ccm, method="union12", nuclear_attraction="program-union")
    with pytest.raises(ValueError, match="nuclear_attraction must be one of"):
        run_ccm_rhf(ccm, method="p2014", nuclear_attraction="nope")
