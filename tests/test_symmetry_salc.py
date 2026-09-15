"""Character tables + irrep matrix projection (``symmetry_salc``).

Regression for the 2026-06-10 fix: ``symmetry_project_matrix`` used to
ignore the per-operator characters entirely (every irrep got the scaled
group average), and the bundled m-3m table had its 8C3↔3C2 and 8S6↔3σh
class rows swapped (the m-3 Th table's ungerade rows were wrong too).

The math being pinned (standard finite-group representation theory —
e.g. Bishop, "Group Theory and Chemistry", Dover 1993, ch. 7):

* row orthogonality with class sizes,
  ``(1/|G|) Σ_c |c| χ_α(c) χ_β(c) = n_α δ_αβ`` with ``n_α = 1`` for a
  genuine real irrep and ``2`` for the merged complex-conjugate pairs
  of the real Th table;
* the projector completeness ``Σ_α M_α = M`` (regular-character
  identity ``Σ_α d_α χ_α(R) = |G| δ_{R,E}``);
* the group average of any matrix is purely totally-symmetric;
* projectors are idempotent and mutually orthogonal.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.symmetry_integrals import symmorphic_operations
from vibeqc.symmetry_salc import (
    _CHARACTER_TABLES,
    _CLASS_SPECS,
    character_table,
    symmetry_project_matrix,
)
from vibeqc.symmetry_scf import build_ao_permutation_cache

ANG2BOHR = 1.0 / 0.529177210903


def _mgo_with_symmetry():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0])],
    )
    vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


@pytest.mark.parametrize("point_group", sorted(_CHARACTER_TABLES.keys()))
def test_character_table_row_orthogonality(point_group):
    """Every bundled table satisfies weighted row orthogonality, with
    norms 1 (genuine irrep) or 2 (Th's merged complex pairs)."""
    _, rows, _ = _CHARACTER_TABLES[point_group]
    sizes = np.array([c for _, c in _CLASS_SPECS[point_group]], dtype=float)
    chars = np.array(rows, dtype=float)  # (n_classes, n_irreps)
    order = sizes.sum()
    gram = (chars.T * sizes) @ chars / order
    off = gram - np.diag(np.diag(gram))
    assert np.abs(off).max() < 1e-12, f"{point_group}: rows not orthogonal"
    diag = np.diag(gram)
    assert np.all(
        (np.abs(diag - 1.0) < 1e-12) | (np.abs(diag - 2.0) < 1e-12)
    ), f"{point_group}: character norms {diag} not in {{1, 2}}"
    # identity column = irrep dimensions, all positive integers
    dims = chars[0]
    assert np.all(dims >= 1) and np.allclose(dims, np.round(dims))
    # Σ_α d_α χ_α(c) = |G| δ_{c,E} / merged-norm — regular character.
    # (With merged rows this still vanishes on every c ≠ E because the
    # merged row is the sum of the pair's characters.)
    reg = chars @ (chars[0] / np.where(np.abs(diag - 2) < 1e-12, 2.0, 1.0))
    assert abs(reg[0] - order) < 1e-9
    assert np.abs(reg[1:]).max() < 1e-9


def test_mgo_oh_class_structure():
    """MgO (Fm-3m → m-3m): conjugacy classes computed from the 48
    symmorphic operators match the Oh table column structure."""
    sysp, _basis = _mgo_with_symmetry()
    ops = symmorphic_operations(sysp.symmetry.operations)
    assert len(ops) == 48
    ct = character_table("m-3m", ops)
    assert ct is not None
    sizes = [len(c) for c in ct.class_indices]
    assert sizes == [1, 8, 3, 6, 6, 1, 8, 3, 6, 6]
    # classes partition the operator list
    flat = sorted(i for cls in ct.class_indices for i in cls)
    assert flat == list(range(48))


def test_project_completeness_and_a1g_purity_mgo():
    """Σ_α M_α = M for arbitrary M; the group average projects purely
    onto A1g; components are idempotent and mutually orthogonal."""
    sysp, basis = _mgo_with_symmetry()
    ops = symmorphic_operations(sysp.symmetry.operations)
    P_cache = build_ao_permutation_cache(sysp, basis, ops)

    rng = np.random.default_rng(7)
    M = rng.standard_normal(P_cache[0].shape)
    M = 0.5 * (M + M.T)

    comps = symmetry_project_matrix(M, sysp, basis)
    assert set(comps) == {
        "A1g", "A2g", "Eg", "T1g", "T2g",
        "A1u", "A2u", "Eu", "T1u", "T2u",
    }
    total = sum(comps.values())
    assert np.abs(total - M).max() < 1e-12, "completeness Σ_α M_α = M"

    # group average → totally symmetric only
    M_avg = sum(P @ M @ P.T for P in P_cache) / len(P_cache)
    avg_comps = symmetry_project_matrix(M_avg, sysp, basis)
    assert np.abs(avg_comps["A1g"] - M_avg).max() < 1e-12
    for label, comp in avg_comps.items():
        if label != "A1g":
            assert np.linalg.norm(comp) < 1e-12, f"{label} leaked"

    # idempotence + mutual orthogonality on a non-trivial component
    eg = comps["Eg"]
    assert np.linalg.norm(eg) > 1e-8, "random M should have Eg weight"
    again = symmetry_project_matrix(eg, sysp, basis)
    assert np.abs(again["Eg"] - eg).max() < 1e-12
    for label, comp in again.items():
        if label != "Eg":
            assert np.linalg.norm(comp) < 1e-12, f"P_{label} P_Eg ≠ 0"


def test_project_abelian_mmm():
    """Abelian D2h (one-atom orthorhombic cell): per-operator χ = ±1,
    eight 1-D irreps, completeness + idempotence."""
    lattice = np.diag([6.0, 7.0, 8.0])
    sysp = vq.PeriodicSystem(3, lattice, [vq.Atom(8, [0.0, 0.0, 0.0])])
    vq.attach_symmetry(sysp)
    assert sysp.symmetry is not None
    pg = sysp.symmetry.point_group
    assert pg == "mmm"
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    ops = symmorphic_operations(sysp.symmetry.operations)
    assert len(ops) == 8

    rng = np.random.default_rng(3)
    M = rng.standard_normal((basis.nbasis, basis.nbasis))
    comps = symmetry_project_matrix(M, sysp, basis)
    assert len(comps) == 8
    total = sum(comps.values())
    assert np.abs(total - M).max() < 1e-12
    for label, comp in comps.items():
        again = symmetry_project_matrix(comp, sysp, basis)
        assert np.abs(again[label] - comp).max() < 1e-12
