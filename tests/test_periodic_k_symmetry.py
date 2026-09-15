"""Validated pieces of the experimental k-star transport module.

(The full IBZ-native transport is open — see the module docstring's
probe record; production paths expand IBZ meshes to the full mesh.)
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    compute_overlap_lattice,
    monkhorst_pack,
    real_space_density_from_kpoints,
)
from vibeqc.pbc_bipole_common import (
    _bloch_sum_blocks,
    s_fold_truncation_drift,
)
from vibeqc.periodic_k_symmetry import (
    KMeshUnfolding,
    density_set_from_k_matrices,
    expand_k_matrices_to_full,
    star_operations,
)

ANG2BOHR = 1.0 / 0.529177210903


def _mgo():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    sysp = vq.PeriodicSystem(
        3, lattice,
        [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])],
    )
    vq.attach_symmetry(sysp)
    return sysp


def test_star_operations_covers_full_mesh_mgo():
    sysp = _mgo()
    km_full = monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)
    km_ibz = monkhorst_pack(sysp, [2, 2, 2], use_symmetry=True)
    smap = star_operations(sysp, km_ibz, km_full)
    n_full = len(list(km_full.kpoints))
    n_ibz = len(list(km_ibz.kpoints))
    assert len(smap.entries) == n_full
    reps = {e[0] for e in smap.entries}
    assert reps == set(range(n_ibz)), "every IBZ rep heads a star"
    for i_rep, op_idx, trev in smap.entries:
        assert 0 <= op_idx < len(smap.operations)
        assert isinstance(trev, bool)


def test_star_operations_rejects_foreign_mesh():
    sysp = _mgo()
    km_ibz = monkhorst_pack(sysp, [2, 2, 2], use_symmetry=True)
    km_other = monkhorst_pack(sysp, [3, 3, 3], use_symmetry=False)
    with pytest.raises(ValueError, match="not in the star"):
        star_operations(sysp, km_ibz, km_other)


def test_kmesh_unfolding_oblique_dim2_reproduces_full_mesh_overlap():
    """An oblique 2D star uses the same B q = k convention as native MP."""
    # PeriodicSystem stores Cartesian lattice vectors as columns. The
    # deliberately nonsymmetric reciprocal matrix makes B q = k distinct
    # from the historical, incorrect B.T q = k conversion.
    lattice = np.array(
        [
            [8.0, 2.0, 0.0],
            [0.0, 7.0, 0.0],
            [0.0, 0.0, 30.0],
        ]
    )
    sysp = vq.PeriodicSystem(
        2,
        lattice,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh_full = monkhorst_pack(sysp, [3, 3, 1], use_symmetry=False)
    kmesh_ibz = monkhorst_pack(sysp, [3, 3, 1], use_symmetry=True)
    assert len(kmesh_full.kpoints) == 9
    assert len(kmesh_ibz.kpoints) == 5

    unfolding = KMeshUnfolding.build(sysp, basis, kmesh_ibz)
    assert unfolding.n_full == 9
    assert len(unfolding.rep_full_index) == 5
    assert any(entry[2] for entry in unfolding.star_map.entries)

    S_at = _overlap_k_builder(sysp, basis, cutoff=8.0)
    S_ibz = [S_at(k) for k in kmesh_ibz.kpoints]
    S_unfolded = unfolding.unfold(S_ibz)
    worst = max(
        float(np.max(np.abs(S_j - S_at(k_j))))
        for S_j, k_j in zip(S_unfolded, kmesh_full.kpoints)
    )
    assert worst < 1e-12, worst


def _overlap_k_builder(sysp, basis, cutoff):
    """Return ``S_at(k)`` = directly-Bloch-summed overlap at cutoff."""
    opts = LatticeSumOptions()
    opts.cutoff_bohr = cutoff
    Slat = compute_overlap_lattice(basis, sysp, opts)
    blocks = [np.asarray(b, dtype=float) for b in Slat.blocks]
    cells = Slat.cells

    def S_at(k):
        return _bloch_sum_blocks(blocks, cells, np.asarray(k, dtype=float))

    return S_at


@pytest.mark.parametrize("mesh,exercises_trev", [([2, 2, 2], False),
                                                 ([3, 3, 3], True)])
def test_expand_k_matrices_reproduces_full_mesh_overlap(mesh, exercises_trev):
    """Phase-aware star transport reproduces the directly-built S(k_full).

    The overlap is absolutely convergent and exactly point-group-symmetric
    in the cutoff→∞ limit, so it isolates the transport from any SCF
    cell-list-asymmetry floor (which is why the 2026-06-10 probe on the
    converged density mis-attributed the residual). Regression for the
    phaseless transport ``P·S(k)·Pᵀ``, which carried an O(1) (~0.78) error
    on every shift-bearing operation — e.g. inversion on rock salt sends
    the anion to its image one cell over. The per-atom Bloch phase
    ``e^{i k·L_a}`` restores it to ~1e-9 at a fold-converged cutoff. The
    [3,3,3] mesh exercises the time-reversal branch (k = −R·k_rep).
    """
    sysp = _mgo()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh_full = monkhorst_pack(sysp, mesh, use_symmetry=False)
    kmesh_ibz = monkhorst_pack(sysp, mesh, use_symmetry=True)
    smap = star_operations(sysp, kmesh_ibz, kmesh_full)
    n_trev = sum(1 for e in smap.entries if e[2])
    assert (n_trev > 0) == exercises_trev

    S_at = _overlap_k_builder(sysp, basis, cutoff=20.0)
    kibz = [np.asarray(k, dtype=float).reshape(3) for k in kmesh_ibz.kpoints]
    kfull = [np.asarray(k, dtype=float).reshape(3) for k in kmesh_full.kpoints]

    S_ibz = [S_at(k) for k in kibz]
    S_expanded = expand_k_matrices_to_full(
        S_ibz, smap, sysp, basis, kmesh_full)

    assert len(S_expanded) == len(kfull)
    worst = 0.0
    for j in range(len(kfull)):
        Mj = S_expanded[j]
        # transported blocks must stay Hermitian
        assert np.max(np.abs(Mj - Mj.conj().T)) < 1e-10
        worst = max(worst, float(np.max(np.abs(Mj - S_at(kfull[j])))))
    assert worst < 1e-6, f"mesh {mesh}: worst transport residual {worst:.3e}"


def test_density_set_from_k_matrices_matches_cxx_builder():
    """The from-D(k) inverse-Bloch fold reproduces the C++
    coefficient-based builder on a full mesh with integer occupations.

    The transform is algebraic and does not require a converged SCF solution.
    Build a physically valid determinant directly from the principal
    ``S(k)^-1/2`` on fold-converged support: its coefficients are
    S-orthonormal, its density is S-idempotent, and the odd mesh exercises
    genuinely complex time-reversal pairs.
    """
    sysp = _mgo()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = monkhorst_pack(sysp, [3, 3, 3], use_symmetry=False)
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 16.0
    lat.nuclear_cutoff_bohr = 16.0
    k_points = [np.asarray(k, dtype=float) for k in km.kpoints]

    drift = s_fold_truncation_drift(
        basis,
        sysp,
        lat,
        k_points=k_points,
    )
    assert drift < 1e-4

    overlap = compute_overlap_lattice(basis, sysp, lat)
    overlap_blocks = [np.asarray(block, dtype=float) for block in overlap.blocks]
    n_occ = sysp.n_electrons() // 2
    coefficients = []
    D_k = []
    electron_count = 0.0
    max_density_imaginary = 0.0
    # Pisani et al., DOI 10.1007/978-3-642-93385-1, Eqs. II.3.2 and
    # II.2.9: form P(k) from occupied coefficients, then inverse-Bloch fold
    # it. That algebra applies at every SCF cycle, not only at convergence.
    # McClain et al., DOI 10.1021/acs.jctc.7b00049, Eqs. 13-14 pin the
    # normalized full-mesh sampling and the closed-shell factor of two.
    for weight, k_point in zip(km.weights, k_points):
        S_k = _bloch_sum_blocks(overlap_blocks, overlap.cells, k_point)
        S_k = 0.5 * (S_k + S_k.conj().T)
        C_k, _ = vq.symmetric_orth(S_k)
        np.testing.assert_allclose(
            C_k.conj().T @ S_k @ C_k,
            np.eye(basis.nbasis),
            atol=1e-12,
            rtol=0.0,
        )
        C_occ = C_k[:, :n_occ]
        density = 2.0 * (C_occ @ C_occ.conj().T)
        np.testing.assert_allclose(
            density @ S_k @ density,
            2.0 * density,
            atol=1e-11,
            rtol=0.0,
        )
        electron_count += float(weight) * float(np.trace(density @ S_k).real)
        max_density_imaginary = max(
            max_density_imaginary,
            float(np.max(np.abs(density.imag))),
        )
        coefficients.append(C_k)
        D_k.append(density)

    assert electron_count == pytest.approx(float(sysp.n_electrons()), abs=1e-10)
    assert max_density_imaginary > 1e-3

    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(sysp.lattice)).T
    fractional_k = [
        np.linalg.solve(reciprocal, k_point) for k_point in k_points
    ]
    for index, q_point in enumerate(fractional_k):
        partners = [
            partner
            for partner, q_partner in enumerate(fractional_k)
            if np.max(
                np.abs(q_point + q_partner - np.rint(q_point + q_partner))
            )
            < 1e-10
        ]
        assert len(partners) == 1
        np.testing.assert_allclose(
            D_k[partners[0]],
            D_k[index].conj(),
            atol=1e-12,
            rtol=0.0,
        )

    ref = real_space_density_from_kpoints(
        coefficients,
        [n_occ] * len(D_k),
        km,
        list(overlap.cells),
    )
    mine = density_set_from_k_matrices(sysp, basis, lat, km, D_k)
    ref_by_cell = {
        tuple(int(x) for x in cell.index): np.asarray(block, dtype=float)
        for cell, block in zip(ref.cells, ref.blocks)
    }
    mine_by_cell = {
        tuple(int(x) for x in cell.index): np.asarray(block, dtype=float)
        for cell, block in zip(mine.cells, mine.blocks)
    }
    assert mine_by_cell.keys() == ref_by_cell.keys()
    assert len(mine_by_cell) >= 5
    for key, block in mine_by_cell.items():
        np.testing.assert_allclose(
            block,
            ref_by_cell[key],
            atol=1e-12,
            rtol=0.0,
        )
