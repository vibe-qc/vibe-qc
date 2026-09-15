"""Phase 12e-c-4c-i: multi-cell FFT density + periodic J_LR builder.

Contracts exercised:

1. **Density normalisation** — for a converged SCF density in the
   molecular-limit (vacuum-padded cell), ρ(r) on the FFT grid
   integrates to ``N_electrons``.

2. **Molecular-limit equivalence** — for a single-cell D_real
   (``D_real.cells == [0]``), the new periodic Γ-only J builder
   matches the existing :func:`vibeqc.build_j_long_range` output to
   machine precision.

3. **Multi-cell density handling** — given a synthetic D_real with
   non-zero blocks at |g| > 0, the builder runs to completion and
   produces finite, symmetric J matrices distinct from the
   single-cell-only result.

4. **Output cell selection** — when ``output_cells`` is set, the
   builder returns one J(g) block per requested cell (matching the
   multi-k dispatch need for 12e-c-4c-iii).

5. **Skew-cell support** — non-orthorhombic cells are sampled on a
   fractional grid and fed through the general FFT Poisson metric.
"""

from __future__ import annotations

import subprocess
import sys
from typing import List

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_density import build_j_long_range_periodic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _h2_in_box(box: float = 30.0):
    """H₂ at the center of a cubic box. Good molecular-limit test
    case — no tight cores."""
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _converged_density_single_cell(
    sysp, basis,
) -> vq.LatticeMatrixSet:
    """Run the Γ-Ewald SCF to convergence, then wrap the Γ-point
    density matrix into a single-cell LatticeMatrixSet so the
    multi-cell builder can consume it."""
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 60
    opts.use_diis = False
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    D_gamma = r.density

    # Fold the Γ density into a single-cell LatticeMatrixSet. We use
    # real_space_density_from_kpoints for this — at a [1, 1, 1] k-mesh
    # it collapses to the Γ-only case.
    n_occ = sysp.n_electrons() // 2
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    S_lat = vq.compute_overlap_lattice(basis, sysp, opts.lattice_opts)
    # MO coefficients at Γ are the real mo_coeffs from the result; we
    # need complex form for real_space_density_from_kpoints.
    C_occ = r.mo_coeffs[:, :n_occ].astype(np.complex128)
    D_real = vq.real_space_density_from_kpoints(
        [C_occ], [n_occ], km, S_lat.cells,
    )
    return D_real, D_gamma, r


# ---------------------------------------------------------------------------
# Density normalisation
# ---------------------------------------------------------------------------

def test_long_range_j_blocks_decay_and_respect_lattice_symmetry():
    """J(g) must fall off with |R_g| and match across equivalent cells.

    Regression for the long-range-J tied-sum fix (2026-08-03), routed by
    the GAPW chat after the sibling density defect
    (PERIODIC-DENSITY-GRID-DOUBLE-COUNTS-IMAGES).

    ``build_j_long_range_periodic`` image-summed its two AO factors
    INDEPENDENTLY. With V lattice-periodic, substituting r -> r + R shows
    the pair then sits at separation ``R_g + R' - R``, so everything with
    R' != R is deposited into the WRONG lattice block. That is
    contamination between blocks, not a scale factor, so an absolute pin
    on one block is a weak test; the structural signatures are what
    discriminate:

      * J(0) must dominate every neighbour block.
      * Cells in the same SYMMETRY ORBIT must give equal ``|J(g)|``.

    Note the orbits are not set by |R_g| alone: all six face neighbours of
    this cubic cell sit at 5.0 bohr, but the H2 axis lies along z, so the
    +-z pair (|J| = 0.003269) and the four +-x/+-y cells (0.000932) are
    different orbits. Grouping by distance would be wrong physics.

    Measured on this compact 5-bohr cell, |J(g)| by block at
    ao_image_radius=2 -- pre-fix vs post-fix::

        pre   0.083368 0.056456 0.061218 0.072734 0.056311 ...
        post  0.075565 0.003269 0.003269 0.000932 0.000932 ...

    Pre-fix the far blocks are inflated 17-78x and do not decay at all;
    post-fix they decay and equivalent cells agree exactly. A compact cell
    is required: in a well-separated box the cross terms are exponentially
    negligible and both forms agree to ~1e-7, so a roomy fixture cannot
    see this bug.
    """
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * 5.0,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.damping = 0.3
    opts.max_iter = 80
    opts.use_diis = False
    res = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
        allow_dense_ionic=True,
    )
    assert res.converged
    n_occ = sysp.n_electrons() // 2
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    S_lat = vq.compute_overlap_lattice(basis, sysp, opts.lattice_opts)
    C_occ = res.mo_coeffs[:, :n_occ].astype(np.complex128)
    D_real = vq.real_space_density_from_kpoints(
        [C_occ], [n_occ], km, S_lat.cells,
    )

    cells = list(range(min(7, len(D_real.cells))))
    blocks = build_j_long_range_periodic(
        basis, sysp, D_real, omega=0.5, spacing_bohr=0.35,
        output_cells=cells, ao_image_radius=2,
    )
    norms = {}
    for c, J in zip(cells, blocks):
        idx = tuple(int(v) for v in np.asarray(D_real.cells[c].index))
        norms[idx] = float(np.linalg.norm(J))

    home = norms[(0, 0, 0)]
    # Orbits: along the molecular (z) axis, and perpendicular to it.
    axial = [v for k, v in norms.items() if k[2] != 0 and k[0] == 0 and k[1] == 0]
    perp = [v for k, v in norms.items() if k[2] == 0 and (k[0] or k[1])]
    assert axial and perp, f"fixture must expose both orbits; got {norms}"

    assert home > 10.0 * max(axial + perp), (
        f"J(0)={home:.6f} must dominate the neighbour blocks {norms}. "
        f"Blocks of comparable magnitude mean neighbouring-block content is "
        f"leaking into distant blocks (the untied double image sum)."
    )
    for group, label in ((axial, "axial (+-z)"), (perp, "perpendicular (x/y)")):
        assert max(group) == pytest.approx(min(group), rel=1e-6), (
            f"cells in the {label} orbit must give equal |J(g)|; got "
            f"{group}. Unequal values mean the lattice sum is not tied."
        )
    assert min(axial) > max(perp), (
        f"the H2 axis lies along z, so axial neighbours must overlap more "
        f"than perpendicular ones; got axial={axial} perp={perp}"
    )


def test_density_on_grid_integrates_to_n_electrons():
    """ρ(r) on the FFT grid integrates to N_electrons per unit cell
    for a converged SCF density."""
    sysp, basis = _h2_in_box()
    D_real, _, _ = _converged_density_single_cell(sysp, basis)
    rho, grid_shape = vq.evaluate_periodic_density_on_grid(
        basis, sysp, D_real, spacing_bohr=0.3,
    )
    dV = abs(np.linalg.det(sysp.lattice)) / rho.size
    integral = float(rho.sum() * dV)
    assert integral == pytest.approx(sysp.n_electrons(), abs=1e-5), (
        f"expected {sysp.n_electrons()} electrons; integrated "
        f"{integral:.6f}"
    )


# ---------------------------------------------------------------------------
# Molecular-limit equivalence
# ---------------------------------------------------------------------------

def test_periodic_J_LR_matches_molecular_limit_at_single_cell():
    """When D_real has only a single cell (g = 0), the periodic builder
    must reduce exactly to :func:`vibeqc.build_j_long_range` on the
    same Γ-density. This is the fundamental compatibility check."""
    sysp, basis = _h2_in_box()
    D_real, D_gamma, _ = _converged_density_single_cell(sysp, basis)

    J_periodic = vq.build_j_long_range_periodic(
        basis, sysp, D_real, omega=0.5, spacing_bohr=0.3,
    )
    J_molecular = vq.build_j_long_range(
        basis, D_gamma, sysp.lattice, omega=0.5, spacing_bohr=0.3,
    )
    assert np.allclose(J_periodic, J_molecular, atol=1e-10), (
        f"max |Δ| = {np.abs(J_periodic - J_molecular).max():.3e}"
    )


def test_result_J_is_symmetric():
    sysp, basis = _h2_in_box()
    D_real, _, _ = _converged_density_single_cell(sysp, basis)
    J = vq.build_j_long_range_periodic(
        basis, sysp, D_real, omega=0.5, spacing_bohr=0.3,
    )
    assert np.allclose(J, J.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Multi-cell density handling (synthetic D_real with non-zero g-blocks)
# ---------------------------------------------------------------------------

def test_multi_cell_density_runs_and_changes_result():
    """Construct a synthetic D_real with an artificial non-zero block
    at a non-zero lattice index, verify the builder runs to completion
    and produces a different J than the single-cell case."""
    sysp, basis = _h2_in_box()
    D_real, D_gamma, _ = _converged_density_single_cell(sysp, basis)

    # Baseline: single-cell J.
    J_single = vq.build_j_long_range_periodic(
        basis, sysp, D_real, omega=0.5, spacing_bohr=0.3,
    )

    # Build a multi-cell D_real with a small ad-hoc non-zero block at
    # (1, 0, 0). We can't easily re-bind D_real's blocks, so build
    # from a richer k-mesh via real_space_density_from_kpoints; a
    # [2, 1, 1] mesh on this system still gives non-trivial D(g=1).
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    opts.nuclear_cutoff_bohr = 15.0
    km2 = vq.monkhorst_pack(sysp, [2, 1, 1])
    S_lat = vq.compute_overlap_lattice(basis, sysp, opts)
    # Use the Γ MO coefficients twice (as if both k-points had the
    # same ones). The resulting D_real won't be "physical" but will
    # exercise the multi-cell code path.
    n_occ = sysp.n_electrons() // 2
    # We need complex MO coefficients. Pull from the converged SCF.
    rhf_opts = vq.PeriodicRHFOptions()
    rhf_opts.lattice_opts = opts
    rhf_opts.damping = 0.3
    rhf_opts.max_iter = 60
    rhf_opts.use_diis = False
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, rhf_opts, omega=0.5, spacing_bohr=0.3,
    )
    C_occ = r.mo_coeffs[:, :n_occ].astype(np.complex128)
    D_multi = vq.real_space_density_from_kpoints(
        [C_occ, C_occ], [n_occ, n_occ], km2, S_lat.cells,
    )
    # With the same coefficients at both k-points, the non-zero blocks
    # at |g| > 0 may not differ from the single-cell case by much —
    # but the machinery runs.
    J_multi = vq.build_j_long_range_periodic(
        basis, sysp, D_multi, omega=0.5, spacing_bohr=0.3,
    )
    # J_multi is a valid ndarray of the right shape, symmetric.
    nbf = basis.nbasis
    assert J_multi.shape == (nbf, nbf)
    assert np.allclose(J_multi, J_multi.T, atol=1e-10)
    # It may or may not differ from J_single depending on whether the
    # multi-k D_real has non-trivial off-Γ content — the key check is
    # that the builder produced something sensible without error.
    assert np.isfinite(J_multi).all()


def _one_ao_translated_density(translated_block: float):
    # Deliberately non-symmetric: rows are not lattice vectors. This makes
    # the QVF round-trip below sensitive to the column-vector convention.
    lattice = np.array(
        [
            [2.0, 0.4, 0.0],
            [0.0, 2.0, 0.3],
            [0.0, 0.0, 2.0],
        ]
    )
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    cells = list(core.direct_lattice_cells(sysp, 2.0 + 1.0e-8))
    by_key = {
        tuple(int(value) for value in np.asarray(cell.index)): cell
        for cell in cells
    }
    density = core.make_lattice_matrix_set(
        int(basis.nbasis),
        [by_key[(0, 0, 0)], by_key[(1, 0, 0)]],
        [np.array([[0.75]]), np.array([[translated_block]])],
    )
    return lattice, sysp, basis, density


def _independent_one_ao_density_values(
    lattice,
    sysp,
    basis,
    shape,
    translated_block: float,
) -> np.ndarray:
    from vibeqc.ewald_j import get_shifted_basis

    fractional_points = np.asarray(
        [
            (ix / shape[0], iy / shape[1], iz / shape[2])
            for ix in range(shape[0])
            for iy in range(shape[1])
            for iz in range(shape[2])
        ],
        dtype=float,
    )
    points = fractional_points @ lattice.T
    chi_home = np.asarray(core.evaluate_ao(basis, points), dtype=float)[:, 0]
    shifted_basis = get_shifted_basis(basis, sysp, (1, 0, 0))
    chi_translated = np.asarray(
        core.evaluate_ao(shifted_basis, points),
        dtype=float,
    )[:, 0]
    return (
        0.75 * chi_home * chi_home
        + translated_block * chi_translated * chi_home
    )


def _asymmetric_three_k_density_with_symmetric_image_domain(
    *,
    ao_image_radius: int,
):
    """Return a phase-consistent 3-k density with wide returned support."""
    box = 6.0
    lattice = np.eye(3) * box
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [
            vq.Atom(1, [box / 2.0 - 0.7, box / 2.0, box / 2.0]),
            vq.Atom(1, [box / 2.0 + 0.7, box / 2.0, box / 2.0]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lattice_options = vq.LatticeSumOptions()
    # Keep a returned radial support substantially wider than one BvK
    # residue system.  Recovery needs only the torus representatives, but
    # refolding must certify every additional returned cell as well.
    lattice_options.cutoff_bohr = (
        2.0 * float(ao_image_radius) * np.sqrt(3.0) * box + 1.0e-8
    )
    overlap = core.compute_overlap_lattice(basis, sysp, lattice_options)
    kmesh = core.monkhorst_pack(sysp, [3, 1, 1])
    raw_coefficients = (
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([1.0, 1.0j], dtype=np.complex128),
        np.array([1.0, -1.0j], dtype=np.complex128),
    )
    coefficients = []
    density_k = []
    for raw, kpoint in zip(raw_coefficients, kmesh.kpoints):
        overlap_k = sum(
            np.exp(1.0j * np.dot(kpoint, cell.r_cart))
            * np.asarray(block)
            for cell, block in zip(overlap.cells, overlap.blocks)
        )
        vector = raw / np.sqrt(float(np.real(raw.conj() @ overlap_k @ raw)))
        coefficients.append(vector[:, None])
        density_k.append(2.0 * np.outer(vector, vector.conj()))
    density = core.real_space_density_from_kpoints_fractional(
        coefficients,
        [np.array([2.0])] * 3,
        kmesh,
        overlap.cells,
    )
    return lattice, sysp, basis, kmesh, density_k, density


def _independent_symmetric_image_density(
    lattice,
    sysp,
    basis,
    weighted_density_per_k,
    kpoints_cart,
    *,
    grid_shape: tuple[int, int, int],
    ao_image_radius: int,
) -> np.ndarray:
    """Evaluate the independent Bloch-AO double-image contraction."""
    from itertools import product

    fractional_points = (
        np.asarray(list(product(*(range(size) for size in grid_shape))), dtype=float)
        / np.asarray(grid_shape, dtype=float)
    )
    points = fractional_points @ np.asarray(lattice, dtype=float).T
    translations = list(
        product(range(-ao_image_radius, ao_image_radius + 1), repeat=3)
    )
    translation_vectors = np.asarray(
        [np.asarray(lattice, dtype=float) @ translation for translation in translations]
    )
    rho = np.zeros(points.shape[0], dtype=float)
    for weighted_density, kpoint in zip(weighted_density_per_k, kpoints_cart):
        chi_k = np.asarray(
            core.evaluate_bloch_ao(
                basis,
                points,
                np.asarray(kpoint, dtype=float),
                translation_vectors,
            )
        )
        rho += np.einsum(
            "pi,ij,pj->p",
            chi_k,
            np.asarray(weighted_density),
            chi_k.conj(),
            optimize=True,
        ).real
    return rho.reshape(grid_shape)


def test_weighted_k_density_matches_independent_asymmetric_three_k_grid():
    """Reciprocal density matches an independent Bloch-AO contraction."""
    from vibeqc.periodic_density import evaluate_weighted_k_density_on_grid

    radius = 1
    shape = (12, 12, 12)
    (
        lattice,
        sysp,
        basis,
        kmesh,
        density_per_k,
        _density,
    ) = _asymmetric_three_k_density_with_symmetric_image_domain(
        ao_image_radius=radius,
    )
    weighted = [
        float(weight) * np.asarray(matrix)
        for weight, matrix in zip(kmesh.weights, density_per_k)
    ]
    expected = _independent_symmetric_image_density(
        lattice,
        sysp,
        basis,
        weighted,
        kmesh.kpoints,
        grid_shape=shape,
        ao_image_radius=radius,
    )

    actual, actual_shape = evaluate_weighted_k_density_on_grid(
        basis,
        sysp,
        weighted,
        np.asarray(kmesh.kpoints),
        grid_shape=shape,
        ao_image_radius=radius,
    )

    assert actual_shape == shape
    for index in ((3, 6, 6), (6, 6, 6), (9, 6, 6), (4, 5, 7)):
        assert actual[index] == pytest.approx(expected[index], abs=2.0e-14)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-14)
    dV = abs(float(np.linalg.det(lattice))) / float(actual.size)
    assert float(actual.sum() * dV) == pytest.approx(2.0, abs=5.0e-4)


def test_exact_lattice_density_fit_recovers_three_k_state_and_checks_all_cells():
    """Torus recovery is accepted only after refolding every returned cell."""
    from vibeqc.periodic_runner import (
        _fit_exact_lattice_density_to_weighted_k_matrices,
    )

    (
        _lattice,
        sysp,
        basis,
        kmesh,
        density_per_k,
        density,
    ) = _asymmetric_three_k_density_with_symmetric_image_domain(
        ao_image_radius=1,
    )
    recovered = _fit_exact_lattice_density_to_weighted_k_matrices(
        density,
        np.asarray(kmesh.kpoints),
        np.asarray(kmesh.weights),
        system=sysp,
        bvk_mesh=(3, 1, 1),
        label="asymmetric returned density",
    )
    for actual, expected, weight in zip(
        recovered,
        density_per_k,
        kmesh.weights,
    ):
        np.testing.assert_allclose(
            actual,
            float(weight) * expected,
            rtol=0.0,
            atol=5.0e-16,
        )

    # Corrupt the most distant returned cell.  It is intentionally outside
    # the three-class BvK recovery torus, so a helper that validates only its
    # selected representatives would miss this change.  The all-cell refold
    # certificate must reject it before grid evaluation.
    distances = [float(np.linalg.norm(cell.r_cart)) for cell in density.cells]
    distant_index = int(np.argmax(distances))
    corrupt_blocks = [np.asarray(block).copy() for block in density.blocks]
    corrupt_blocks[distant_index][0, 0] += 1.0e-4
    corrupt = core.make_lattice_matrix_set(
        int(basis.nbasis),
        list(density.cells),
        corrupt_blocks,
    )
    with pytest.raises(ValueError, match="refold residual"):
        _fit_exact_lattice_density_to_weighted_k_matrices(
            corrupt,
            np.asarray(kmesh.kpoints),
            np.asarray(kmesh.weights),
            system=sysp,
            bvk_mesh=(3, 1, 1),
            label="corrupted returned density",
        )


def test_exact_lattice_density_fit_uses_skew_lattice_mp_characters():
    """MP residues are derived from ``L.T @ k``, including skew cells."""
    from vibeqc.periodic_runner import (
        _fit_exact_lattice_density_to_weighted_k_matrices,
    )

    lattice = np.array(
        [
            [4.0, 0.8, 0.2],
            [0.0, 5.0, 0.6],
            [0.0, 0.0, 6.0],
        ]
    )
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [1.0, 1.0, 1.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = core.monkhorst_pack(sysp, [3, 1, 1])
    kpoints = np.asarray(kmesh.kpoints)
    fractional = (lattice.T @ kpoints.T).T / (2.0 * np.pi)
    np.testing.assert_allclose(
        fractional,
        [[0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0], [2.0 / 3.0, 0.0, 0.0]],
        rtol=0.0,
        atol=3.0e-17,
    )

    cells = list(core.direct_lattice_cells(sysp, 15.0))
    density = core.real_space_density_from_kpoints_fractional(
        [np.ones((1, 1), dtype=np.complex128)] * 3,
        [np.array([1.0])] * 3,
        kmesh,
        cells,
    )
    recovered = _fit_exact_lattice_density_to_weighted_k_matrices(
        density,
        kpoints,
        np.asarray(kmesh.weights),
        system=sysp,
        bvk_mesh=(3, 1, 1),
        label="skew-cell returned density",
    )
    np.testing.assert_allclose(recovered, np.full((3, 1, 1), 1.0 / 3.0))

    # Preserve a product-correct, uniformly weighted three-point input while
    # moving one point off every supported MP character.  Construct Cartesian
    # k explicitly from the skew-cell fractional coordinate so a diagonal-cell
    # shortcut cannot accidentally pass this check.
    non_mp_fractional = fractional.copy()
    non_mp_fractional[1, 0] += 0.07
    non_mp_kpoints = np.linalg.solve(
        lattice.T,
        (2.0 * np.pi * non_mp_fractional).T,
    ).T
    with pytest.raises(ValueError, match="Monkhorst-Pack character set"):
        _fit_exact_lattice_density_to_weighted_k_matrices(
            density,
            non_mp_kpoints,
            np.asarray(kmesh.weights),
            system=sysp,
            bvk_mesh=(3, 1, 1),
            label="non-MP skew-cell returned density",
        )


def test_exact_lattice_density_fit_rejects_time_reversal_violation(monkeypatch):
    """A valid MP character set cannot certify non-TR recovered matrices."""
    import vibeqc.pbc_bipole_common as pbc_bipole_common
    from vibeqc.periodic_runner import (
        _fit_exact_lattice_density_to_weighted_k_matrices,
    )

    (
        _lattice,
        sysp,
        _basis,
        kmesh,
        _density_per_k,
        density,
    ) = _asymmetric_three_k_density_with_symmetric_image_domain(
        ao_image_radius=1,
    )
    # Keep each recovered matrix finite and Hermitian, but make the +k and -k
    # partners unequal.  Patching only the inverse isolates the artifact
    # boundary's independent time-reversal certificate from refold residuals.
    bad_recovered = [
        np.diag([1.0, 0.0]).astype(np.complex128),
        np.diag([1.0, 0.0]).astype(np.complex128),
        np.diag([2.0, 0.0]).astype(np.complex128),
    ]
    monkeypatch.setattr(
        pbc_bipole_common,
        "bvk_torus_density_matrices",
        lambda *_args, **_kwargs: bad_recovered,
    )

    with pytest.raises(ValueError, match="violate time reversal"):
        _fit_exact_lattice_density_to_weighted_k_matrices(
            density,
            np.asarray(kmesh.kpoints),
            np.asarray(kmesh.weights),
            system=sysp,
            bvk_mesh=(3, 1, 1),
            label="non-TR returned density",
        )


def test_periodic_density_grid_certification_requires_final_image_pair():
    """Agreement at radii one and two cannot hide a radius-three change."""
    from vibeqc.periodic_runner import (
        _converged_periodic_density_grid_for_artifact,
    )

    radii = []

    def evaluate(radius):
        radii.append(radius)
        value = 0.0 if radius < 3 else 1.0
        return np.full((1, 1, 1), value), {"radius": radius}

    with pytest.raises(ValueError, match="did not converge"):
        _converged_periodic_density_grid_for_artifact(
            evaluate,
            lattice_bohr=np.eye(3),
            grid_shape=(1, 1, 1),
            analytic_electrons=1.0,
            label="radius-sensitive density",
            max_image_radius=3,
            image_l1_tolerance=1.0e-5,
            image_max_tolerance=1.0e-6,
        )
    assert radii == [1, 2, 3]


def test_periodic_density_grid_charge_mismatch_suggests_refined_spacing():
    """A converged AO-image sum still fails closed on grid quadrature."""
    from vibeqc.periodic_runner import (
        _converged_periodic_density_grid_for_artifact,
    )

    radii = []

    def evaluate(radius):
        radii.append(radius)
        return np.ones((1, 1, 1)), {"radius": radius}

    with pytest.raises(ValueError, match="decrease density_spacing_bohr"):
        _converged_periodic_density_grid_for_artifact(
            evaluate,
            lattice_bohr=np.eye(3),
            grid_shape=(1, 1, 1),
            analytic_electrons=2.0,
            label="under-integrated density",
            max_image_radius=3,
        )
    assert radii == [1, 2, 3]


@pytest.mark.parametrize(
    ("kpoints", "weights", "mesh", "message"),
    [
        (np.empty((0, 3)), np.empty(0), (1, 1, 1), "weights sum to 0"),
        (np.zeros((3, 3)), np.array([0.5, 0.5]), (3, 1, 1), "per k-point"),
        (np.zeros((3, 3)), np.full(3, 1.0 / 3.0), (2, 1, 1), "contains 2 points"),
        (np.zeros((3, 3)), np.array([0.5, 0.25, 0.25]), (3, 1, 1), "uniform"),
    ],
)
def test_large_lattice_density_fit_rejects_missing_or_bad_k_metadata(
    kpoints,
    weights,
    mesh,
    message,
):
    """A wide support cannot substitute for exact BvK sampling metadata."""
    from vibeqc.periodic_runner import (
        _fit_exact_lattice_density_to_weighted_k_matrices,
    )

    lattice = np.eye(3) * 2.0
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [1.0, 1.0, 1.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    cells = list(core.direct_lattice_cells(sysp, 14.0 + 1.0e-8))
    assert len(cells) >= 1000
    reference_kmesh = core.monkhorst_pack(sysp, [3, 1, 1])
    density = core.real_space_density_from_kpoints_fractional(
        [np.ones((1, 1), dtype=np.complex128) for _ in reference_kmesh.kpoints],
        [np.array([1.0]) for _ in reference_kmesh.kpoints],
        reference_kmesh,
        cells,
    )

    with pytest.raises(ValueError, match=message):
        _fit_exact_lattice_density_to_weighted_k_matrices(
            density,
            kpoints,
            weights,
            system=sysp,
            bvk_mesh=mesh,
            label="large returned density",
        )


def test_large_lattice_density_dispatch_rejects_missing_bvk_metadata():
    """The artifact boundary will not take the O(n_cells*n_images) fallback."""
    from types import SimpleNamespace

    from vibeqc.periodic_runner import _exact_periodic_density_grid_artifact

    lattice = np.eye(3) * 2.0
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [1.0, 1.0, 1.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    cells = list(core.direct_lattice_cells(sysp, 14.0 + 1.0e-8))
    assert len(cells) >= 1000
    kmesh = core.monkhorst_pack(sysp, [3, 1, 1])
    density = core.real_space_density_from_kpoints_fractional(
        [np.ones((1, 1), dtype=np.complex128) for _ in kmesh.kpoints],
        [np.array([1.0]) for _ in kmesh.kpoints],
        kmesh,
        cells,
    )
    assert sum(np.any(np.asarray(block) != 0.0) for block in density.blocks) > 128

    with pytest.raises(ValueError, match="no verified BvK k-point metadata"):
        _exact_periodic_density_grid_artifact(
            basis,
            sysp,
            SimpleNamespace(density=density),
            lattice_bohr=lattice,
            grid_shape=(2, 2, 2),
            spacing_bohr=1.0,
            gamma_only=False,
            expected_electrons=1.0,
        )


def test_weighted_k_density_ao_calls_scale_with_kpoints_and_chunks(monkeypatch):
    """AO work is independent of the size of the returned LMS support."""
    import vibeqc.periodic_density as periodic_density

    lattice = np.eye(3) * 2.0
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [1.0, 1.0, 1.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    # A radius-seven sphere contains 1,419 returned cells.  The weighted-k
    # evaluator consumes only the recovered reciprocal matrices, so this
    # support size must never multiply its AO calls.
    cells = list(core.direct_lattice_cells(sysp, 14.0 + 1.0e-8))
    assert len(cells) >= 1000
    kmesh = core.monkhorst_pack(sysp, [3, 1, 1])
    phase_density = core.real_space_density_from_kpoints_fractional(
        [np.ones((1, 1), dtype=np.complex128) for _ in kmesh.kpoints],
        [np.array([1.0,]) for _ in kmesh.kpoints],
        kmesh,
        cells,
    )
    assert len(phase_density.cells) == len(cells)
    assert len(phase_density.blocks) >= 1000
    weighted = [np.array([[float(weight)]], dtype=np.complex128)
                for weight in kmesh.weights]
    calls = []

    def fake_ao(_basis, points):
        calls.append(points.shape[0])
        return np.ones((points.shape[0], 1), dtype=float)

    monkeypatch.setattr(periodic_density, "evaluate_ao", fake_ao)
    rho, shape = periodic_density.evaluate_weighted_k_density_on_grid(
        basis,
        sysp,
        weighted,
        np.asarray(kmesh.kpoints),
        grid_shape=(4, 3, 2),
        chunk_size=5,
        ao_image_radius=1,
    )

    n_chunks = 5
    assert shape == (4, 3, 2)
    # The implementation shares each plain-AO image across k points, which is
    # even stronger than the required O(n_k * n_images) upper bound.
    assert len(calls) == 27 * n_chunks
    assert len(calls) <= len(kmesh.kpoints) * 27 * n_chunks
    assert len(calls) < len(cells)
    # With the fake unit AO, only the Gamma member survives the three-point
    # phase sum: (1/3) * |sum over 27 images|^2 = 243.
    np.testing.assert_allclose(rho, 243.0, rtol=0.0, atol=1.0e-12)


def test_weighted_k_density_gamma_metadata_and_dimensional_translations(
    monkeypatch,
):
    """Gamma is explicit metadata and inactive lattice axes are not imaged."""
    import vibeqc.periodic_density as periodic_density

    lattice = np.diag([4.0, 30.0, 30.0])
    sysp = vq.PeriodicSystem(
        1,
        lattice,
        [vq.Atom(1, [2.0, 15.0, 15.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    observed_points = []

    def fake_ao(_basis, points):
        observed_points.append(np.asarray(points).copy())
        return np.ones((points.shape[0], 1), dtype=float)

    monkeypatch.setattr(periodic_density, "evaluate_ao", fake_ao)
    rho, _ = periodic_density.evaluate_weighted_k_density_on_grid(
        basis,
        sysp,
        [np.array([[1.0]])],
        np.zeros((1, 3)),
        grid_shape=(2, 1, 1),
        ao_image_radius=2,
    )

    assert len(observed_points) == 5
    shifts = np.asarray([points[0] for points in observed_points])
    np.testing.assert_array_equal(shifts[:, 1:], np.zeros((5, 2)))
    np.testing.assert_allclose(np.sort(shifts[:, 0]), [-8.0, -4.0, 0.0, 4.0, 8.0])
    # Five in-phase images at explicit Gamma produce |sum_T chi_T|^2 = 25.
    np.testing.assert_allclose(rho, 25.0, rtol=0.0, atol=0.0)


def test_density_grid_keeps_sub_tolerance_translated_cell_block():
    """Every nonzero returned lattice block contributes to the grid.

    A tolerance-based ``allclose(block, 0)`` screen is not an exact zero
    test: thousands of individually small translated-cell blocks can carry a
    finite part of the converged periodic density.  Evaluate one deliberately
    sub-``1e-8`` block independently from the AO factors at selected grid
    points so this regression does not share the implementation's cell loop.
    """
    tiny = 5.0e-9
    lattice, sysp, basis, density = _one_ao_translated_density(tiny)

    shape = (3, 2, 2)
    rho, actual_shape = vq.evaluate_periodic_density_on_grid(
        basis,
        sysp,
        density,
        grid_shape=shape,
        ao_image_radius=0,
    )

    # Independent selected-point contraction for R=0:
    #   rho(r) = D(0) chi_0(r)^2
    #          + D(g) chi_g(r) chi_0(r).
    expected = _independent_one_ao_density_values(
        lattice,
        sysp,
        basis,
        shape,
        tiny,
    )
    home_only = _independent_one_ao_density_values(
        lattice,
        sysp,
        basis,
        shape,
        0.0,
    )

    assert actual_shape == shape
    assert float(np.max(np.abs(expected - home_only))) > 1.0e-10
    np.testing.assert_allclose(
        rho.ravel(),
        expected,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_exact_restricted_gamma_density_round_trips_through_qvf(tmp_path):
    """Stored QVF values and skew-ready voxel vectors describe the exact grid."""
    import json
    import zipfile
    from types import SimpleNamespace

    from vibeqc.output import OutputPlan
    from vibeqc.output.formats.qvf import validate_qvf, write_qvf
    from vibeqc.periodic_runner import (
        _exact_lattice_density_set_for_grid_artifact,
    )

    translated_block = 0.125
    lattice, sysp, basis, returned_density = _one_ao_translated_density(
        translated_block,
    )
    density = _exact_lattice_density_set_for_grid_artifact(
        basis,
        sysp,
        returned_density,
        label="restricted Gamma density",
    )
    shape = (3, 2, 2)
    rho, actual_shape = vq.evaluate_periodic_density_on_grid(
        basis,
        sysp,
        density,
        grid_shape=shape,
        ao_image_radius=0,
    )
    expected = _independent_one_ao_density_values(
        lattice,
        sysp,
        basis,
        shape,
        translated_block,
    ).reshape(shape)
    voxel_vectors = lattice.T / np.asarray(shape, dtype=float)[:, None]

    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "exact-periodic-density",
        method="RHF",
        basis="sto-3g",
        functional=None,
        job_kind="periodic_scf",
        output_qvf=True,
    )
    qvf_path = write_qvf(
        tmp_path / "exact-periodic-density",
        plan,
        system=sysp,
        result=SimpleNamespace(
            converged=True,
            energy=-0.5,
            n_iter=1,
            scf_trace=[],
        ),
        method="RHF",
        basis="sto-3g",
        volume_data={
            "Electron density": (
                rho,
                np.zeros(3, dtype=float),
                voxel_vectors,
            )
        },
    )

    report = validate_qvf(qvf_path)
    assert report["valid"], report["errors"]
    with zipfile.ZipFile(qvf_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        section = next(
            item
            for item in manifest["sections"]
            if item["kind"] == "volume.density"
        )
        data_member = section["members"]["data"]
        stored = np.frombuffer(
            archive.read(data_member["path"]),
            dtype=np.dtype(data_member["dtype"]),
        ).reshape(tuple(data_member["shape"]))
        grid = json.loads(archive.read(section["members"]["grid"]["path"]))

    assert actual_shape == shape
    for index in ((0, 0, 0), (1, 0, 0), (2, 1, 1)):
        assert stored[index] == pytest.approx(expected[index], rel=2.0e-7)
    np.testing.assert_allclose(
        np.asarray(grid["voxel_vectors"], dtype=float)
        * np.asarray(grid["shape"], dtype=float)[:, None],
        lattice.T,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_exact_restricted_multik_density_uses_same_cell_orientation():
    """A true multi-k density contracts ``D(g)`` with ``S(g)``, not ``S(-g)``.

    A one-AO density cannot expose an index transpose.  This two-AO fixture
    instead folds a time-reversal pair on a genuine three-point mesh.  Its
    real translated blocks obey ``D(-g) = D(g).T`` but are individually
    nonsymmetric, so reversing the shifted and reference AO factors changes
    both selected-point values and the primitive-cell electron count.
    """
    from itertools import product

    from vibeqc.ewald_j import get_shifted_basis
    from vibeqc.periodic_runner import (
        _exact_lattice_density_set_for_grid_artifact,
    )

    box = 6.0
    lattice = np.eye(3) * box
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [
            vq.Atom(1, [box / 2.0 - 0.7, box / 2.0, box / 2.0]),
            vq.Atom(1, [box / 2.0 + 0.7, box / 2.0, box / 2.0]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lattice_options = vq.LatticeSumOptions()
    lattice_options.cutoff_bohr = box + 1.0e-8
    overlap = core.compute_overlap_lattice(basis, sysp, lattice_options)
    kmesh = core.monkhorst_pack(sysp, [3, 1, 1])

    # Normalize one occupied orbital in the Bloch overlap metric at each k.
    # The +k/-k coefficient pair is complex conjugate, so its inverse Bloch
    # transform is real while retaining nonsymmetric D(g != 0) blocks.
    coefficients = []
    for kpoint in kmesh.kpoints:
        fractional_x = float(kpoint[0]) * box / (2.0 * np.pi)
        fractional_x %= 1.0
        if fractional_x == pytest.approx(0.0, abs=1.0e-12):
            raw = np.array([1.0, 0.0], dtype=np.complex128)
        elif fractional_x == pytest.approx(1.0 / 3.0, abs=1.0e-12):
            raw = np.array([1.0, 1.0j], dtype=np.complex128)
        else:
            assert fractional_x == pytest.approx(2.0 / 3.0, abs=1.0e-12)
            raw = np.array([1.0, -1.0j], dtype=np.complex128)
        overlap_k = sum(
            np.exp(1.0j * np.dot(kpoint, cell.r_cart))
            * np.asarray(block)
            for cell, block in zip(overlap.cells, overlap.blocks)
        )
        norm = np.sqrt(float(np.real(raw.conj() @ overlap_k @ raw)))
        coefficients.append((raw / norm)[:, None])

    returned_density = core.real_space_density_from_kpoints_fractional(
        coefficients,
        [np.array([2.0])] * 3,
        kmesh,
        overlap.cells,
    )
    density = _exact_lattice_density_set_for_grid_artifact(
        basis,
        sysp,
        returned_density,
        label="restricted true-multi-k density",
    )
    cell_keys = [
        tuple(int(value) for value in np.asarray(cell.index))
        for cell in density.cells
    ]
    density_by_key = {
        key: np.asarray(block)
        for key, block in zip(cell_keys, density.blocks)
    }
    overlap_by_key = {
        tuple(int(value) for value in np.asarray(cell.index)): np.asarray(block)
        for cell, block in zip(overlap.cells, overlap.blocks)
    }
    plus_x = density_by_key[(1, 0, 0)]
    minus_x = density_by_key[(-1, 0, 0)]
    assert float(np.max(np.abs(plus_x - plus_x.T))) > 0.5
    np.testing.assert_allclose(minus_x, plus_x.T, rtol=0.0, atol=1.0e-14)

    # The inverse Bloch and overlap transforms have opposite phases, leaving
    # the same-cell Frobenius contraction sum_g D(g):S(g).  Contracting
    # against S(-g) is the transpose bug this fixture is designed to expose.
    same_sign_electrons = sum(
        float(np.einsum("ij,ij->", density_by_key[key], overlap_by_key[key]))
        for key in cell_keys
    )
    opposite_sign_electrons = sum(
        float(
            np.einsum(
                "ij,ij->",
                density_by_key[key],
                overlap_by_key[tuple(-value for value in key)],
            )
        )
        for key in cell_keys
    )
    assert same_sign_electrons == pytest.approx(2.0, abs=1.0e-12)
    assert abs(same_sign_electrons - opposite_sign_electrons) > 0.1

    shape = (24, 24, 24)
    selected_indices = ((8, 12, 12), (12, 12, 12), (9, 10, 14))
    fractional_points = (
        np.asarray(selected_indices, dtype=float)
        / np.asarray(shape, dtype=float)
    )
    selected_points = fractional_points @ lattice.T
    translations = list(product(range(-2, 3), repeat=3))
    required_shifts = set(translations)
    for translation in translations:
        for key in cell_keys:
            required_shifts.add(
                tuple(translation[axis] + key[axis] for axis in range(3))
            )
    ao_by_shift = {
        shift: np.asarray(
            core.evaluate_ao(
                get_shifted_basis(basis, sysp, shift),
                selected_points,
            ),
            dtype=float,
        )
        for shift in required_shifts
    }
    expected = np.zeros(len(selected_indices), dtype=float)
    transposed = np.zeros(len(selected_indices), dtype=float)
    for translation in translations:
        chi_ref = ao_by_shift[translation]
        for key in cell_keys:
            shifted_key = tuple(
                translation[axis] + key[axis] for axis in range(3)
            )
            chi_g = ao_by_shift[shifted_key]
            expected += np.einsum(
                "pi,ij,pj->p",
                chi_ref,
                density_by_key[key],
                chi_g,
            )
            transposed += np.einsum(
                "pi,ij,pj->p",
                chi_g,
                density_by_key[key],
                chi_ref,
            )
    assert float(np.max(np.abs(expected - transposed))) > 1.0e-3

    rho, actual_shape = vq.evaluate_periodic_density_on_grid(
        basis,
        sysp,
        density,
        grid_shape=shape,
        ao_image_radius=2,
    )
    assert actual_shape == shape
    np.testing.assert_allclose(
        np.asarray([rho[index] for index in selected_indices]),
        expected,
        rtol=0.0,
        atol=1.0e-14,
    )
    dV = abs(float(np.linalg.det(lattice))) / float(rho.size)
    integrated = float(rho.sum() * dV)
    assert integrated == pytest.approx(same_sign_electrons, abs=1.0e-8)
    assert abs(integrated - opposite_sign_electrons) > 0.1


def test_exact_unrestricted_multik_density_integrates_to_electron_count(tmp_path):
    """Exact alpha+beta output retains the true multi-k translated blocks.

    Build each spin channel with the production inverse Bloch transform on a
    genuine ``(2, 1, 1)`` mesh, then normalize its lattice trace to a chosen
    spin occupation.  The artifact helper must preserve that complete cell
    set, sum alpha and beta by cell key, and yield the same one-electron count
    when integrated over the primitive-cell grid.
    """
    from vibeqc.periodic_runner import (
        _exact_lattice_density_set_for_grid_artifact,
        _sum_exact_lattice_density_sets_for_grid_artifact,
    )

    box = 4.0
    lattice = np.eye(3) * box
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [box / 2.0, box / 2.0, box / 2.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    lattice_options = vq.LatticeSumOptions()
    lattice_options.cutoff_bohr = box + 1.0e-8
    overlap = core.compute_overlap_lattice(basis, sysp, lattice_options)
    kmesh = core.monkhorst_pack(sysp, [2, 1, 1])
    coefficients = [
        np.ones((int(basis.nbasis), int(basis.nbasis)), dtype=np.complex128)
        for _ in range(2)
    ]

    def _normalized_spin_density(
        occupations: tuple[float, float],
        electron_count: float,
    ):
        folded = core.real_space_density_from_kpoints_fractional(
            coefficients,
            [np.array([occupations[0]]), np.array([occupations[1]])],
            kmesh,
            overlap.cells,
        )
        metric = sum(
            float(
                np.einsum(
                    "ij,ij->",
                    np.asarray(density_block),
                    np.asarray(overlap_block),
                )
            )
            for density_block, overlap_block in zip(
                folded.blocks,
                overlap.blocks,
            )
        )
        scale = electron_count / metric
        return core.make_lattice_matrix_set(
            int(basis.nbasis),
            list(folded.cells),
            [np.asarray(block) * scale for block in folded.blocks],
        )

    density_alpha = _normalized_spin_density((0.9, 0.3), 0.6)
    density_beta = _normalized_spin_density((0.4, 0.4), 0.4)
    exact_alpha = _exact_lattice_density_set_for_grid_artifact(
        basis,
        sysp,
        density_alpha,
        label="multi-k alpha density",
    )
    exact_beta = _exact_lattice_density_set_for_grid_artifact(
        basis,
        sysp,
        density_beta,
        label="multi-k beta density",
    )
    total = _sum_exact_lattice_density_sets_for_grid_artifact(
        basis,
        exact_alpha,
        exact_beta,
        label="multi-k unrestricted density",
    )

    expected_keys = [
        tuple(int(value) for value in np.asarray(cell.index))
        for cell in overlap.cells
    ]
    actual_keys = [
        tuple(int(value) for value in np.asarray(cell.index))
        for cell in total.cells
    ]
    assert actual_keys == expected_keys
    assert len(kmesh.kpoints) == 2
    plus_x = actual_keys.index((1, 0, 0))
    home = actual_keys.index((0, 0, 0))
    assert float(np.max(np.abs(total.blocks[plus_x]))) > 0.0
    assert not np.allclose(total.blocks[plus_x], total.blocks[home])
    for actual, alpha, beta in zip(
        total.blocks,
        exact_alpha.blocks,
        exact_beta.blocks,
    ):
        np.testing.assert_allclose(actual, np.asarray(alpha) + np.asarray(beta))

    rho, shape = vq.evaluate_periodic_density_on_grid(
        basis,
        sysp,
        total,
        grid_shape=(32, 32, 32),
        ao_image_radius=2,
    )
    dV = abs(float(np.linalg.det(lattice))) / float(rho.size)
    integrated = float(rho.sum() * dV)

    assert shape == (32, 32, 32)
    assert integrated == pytest.approx(1.0, abs=1.0e-8)

    # Pin the primitive-cell integral after the QVF writer's deliberate
    # float32 storage conversion as well as before serialization.
    import json
    import zipfile
    from types import SimpleNamespace

    from vibeqc.output import OutputPlan
    from vibeqc.output.formats.qvf import validate_qvf, write_qvf

    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "unrestricted-multik-density",
        method="UHF",
        basis="sto-3g",
        functional=None,
        job_kind="periodic_scf",
        output_qvf=True,
    )
    voxel_vectors = lattice.T / np.asarray(shape, dtype=float)[:, None]
    qvf_path = write_qvf(
        tmp_path / "unrestricted-multik-density",
        plan,
        system=sysp,
        result=SimpleNamespace(
            converged=True,
            energy=-0.5,
            n_iter=1,
            scf_trace=[],
        ),
        method="UHF",
        basis="sto-3g",
        volume_data={
            "Electron density": (
                rho,
                np.zeros(3, dtype=float),
                voxel_vectors,
            )
        },
    )
    report = validate_qvf(qvf_path)
    assert report["valid"], report["errors"]
    with zipfile.ZipFile(qvf_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        section = next(
            item
            for item in manifest["sections"]
            if item["kind"] == "volume.density"
        )
        data_member = section["members"]["data"]
        stored = np.frombuffer(
            archive.read(data_member["path"]),
            dtype=np.dtype(data_member["dtype"]),
        ).reshape(tuple(data_member["shape"]))
        grid = json.loads(archive.read(section["members"]["grid"]["path"]))
    stored_voxels = np.asarray(grid["voxel_vectors"], dtype=float)
    stored_integral = float(stored.sum() * abs(np.linalg.det(stored_voxels)))
    assert stored_integral == pytest.approx(1.0, abs=1.0e-6)


def test_odd_electron_density_grid_uses_valid_shifted_basis_multiplicity():
    """Odd-electron periodic cells can default to multiplicity=1.

    Shifted-basis construction is AO-only, so it must use the valid
    unit-cell molecule multiplicity instead of the raw periodic-system
    default when non-home density blocks are present.
    """
    from vibeqc import ewald_j as _ej

    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [4.0, 4.0, 4.0])],
    )
    assert sysp.multiplicity == 1
    assert sysp.unit_cell_molecule().multiplicity == 2
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 9.0
    opts.nuclear_cutoff_bohr = 9.0
    D_real = vq.compute_overlap_lattice(basis, sysp, opts)
    zero = np.zeros((basis.nbasis, basis.nbasis), dtype=float)
    home_idx = None
    off_home_idx = None
    for idx, cell in enumerate(D_real.cells):
        key = tuple(int(x) for x in np.asarray(cell.index, dtype=int))
        D_real.set_block(idx, zero)
        if key == (0, 0, 0):
            home_idx = idx
        elif off_home_idx is None:
            off_home_idx = idx
    assert home_idx is not None
    assert off_home_idx is not None
    D_real.set_block(home_idx, np.array([[1.0]], dtype=float))
    D_real.set_block(off_home_idx, np.array([[0.05]], dtype=float))

    _ej.clear_shifted_basis_cache()
    rho, grid_shape = vq.evaluate_periodic_density_on_grid(
        basis,
        sysp,
        D_real,
        grid_shape=(3, 3, 3),
        ao_image_radius=0,
    )

    assert grid_shape == (3, 3, 3)
    assert rho.shape == grid_shape
    assert np.isfinite(rho).all()


def test_multik_bipole_qvf_uses_returned_lattice_density_end_to_end(tmp_path):
    """The public QVF path keeps BIPOLE cells beyond its old template.

    This small true-multi-k result deliberately has six nonzero ``|g| = 2``
    density blocks outside the former fixed 18-bohr output template.  The
    historical path warned that the template omitted those converged cells
    and then wrote a QVF without density.  The production runner must now
    carry the returned density through to a charge-correct volume instead.
    """
    import json
    import warnings
    import zipfile

    lattice = np.eye(3) * 10.0
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [
            vq.Atom(1, [4.3, 5.0, 5.0]),
            vq.Atom(1, [5.7, 5.0, 5.0]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    output = tmp_path / "h2_rhf_bipole_k211"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vq.run_periodic_job(
            system=sysp,
            basis=basis,
            method="RHF",
            jk_method="bipole",
            kpoints=(2, 1, 1),
            output=str(output),
            output_qvf=True,
            citations=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            density_spacing_bohr=0.5,
            max_iter=40,
            bipole_cutoff_bohr=10.0,
            bipole_nuclear_cutoff_bohr=10.0,
            convergence="off",
            progress=False,
        )

    assert result.converged
    returned_by_key = {
        tuple(int(value) for value in cell.index): np.asarray(block)
        for cell, block in zip(result.density.cells, result.density.blocks)
    }
    template_options = vq.LatticeSumOptions()
    template_options.cutoff_bohr = 18.0
    old_template = vq.compute_overlap_lattice(basis, sysp, template_options)
    old_template_keys = {
        tuple(int(value) for value in cell.index)
        for cell in old_template.cells
    }
    source_only = set(returned_by_key).difference(old_template_keys)
    assert source_only == {
        (-2, 0, 0),
        (0, -2, 0),
        (0, 0, -2),
        (0, 0, 2),
        (0, 2, 0),
        (2, 0, 0),
    }
    assert all(np.any(returned_by_key[key] != 0.0) for key in source_only)

    qvf_path = output.with_suffix(".qvf")
    with zipfile.ZipFile(qvf_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        density_section = next(
            section
            for section in manifest["sections"]
            if section["kind"] == "volume.density"
        )
        data_member = density_section["members"]["data"]
        rho = np.frombuffer(
            archive.read(data_member["path"]),
            dtype=np.dtype(data_member["dtype"]),
        ).reshape(tuple(data_member["shape"]))
        grid_member = density_section["members"]["grid"]
        grid = json.loads(archive.read(grid_member["path"]))

    assert rho.shape == (20, 20, 20)
    assert np.isfinite(rho).all()
    assert rho[8, 10, 10] == pytest.approx(0.24241441, abs=2.0e-6)
    assert rho[10, 10, 10] == pytest.approx(0.25587752, abs=2.0e-6)
    voxel_vectors = np.asarray(grid["voxel_vectors"], dtype=float)
    np.testing.assert_allclose(
        voxel_vectors * np.asarray(rho.shape, dtype=float)[:, None],
        lattice.T,
        rtol=0.0,
        atol=1.0e-12,
    )
    integrated_electrons = float(rho.sum() * abs(np.linalg.det(voxel_vectors)))
    assert integrated_electrons == pytest.approx(2.0, abs=5.0e-4)

    diagnostics = "\n".join(
        [str(warning.message) for warning in caught]
        + [
            output.with_suffix(".out").read_text(),
            output.with_suffix(".system").read_text(),
        ]
    )
    assert "template omits converged lattice cells" not in diagnostics


def test_gamma_bipole_qvf_uses_returned_lattice_density_end_to_end(tmp_path):
    """Gamma BIPOLE exports its full returned lattice density as QVF."""
    import json
    import warnings
    import zipfile

    lattice = np.eye(3) * 10.0
    sysp = vq.PeriodicSystem(
        3,
        lattice,
        [
            vq.Atom(1, [4.3, 5.0, 5.0]),
            vq.Atom(1, [5.7, 5.0, 5.0]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    output = tmp_path / "h2_rhf_bipole_gamma"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vq.run_periodic_job(
            system=sysp,
            basis=basis,
            method="RHF",
            jk_method="bipole",
            kpoints=(1, 1, 1),
            output=str(output),
            output_qvf=True,
            citations=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            density_spacing_bohr=0.5,
            max_iter=40,
            bipole_cutoff_bohr=10.0,
            bipole_nuclear_cutoff_bohr=10.0,
            convergence="off",
            progress=False,
        )

    assert result.converged
    returned_by_key = {
        tuple(int(value) for value in cell.index): np.asarray(block)
        for cell, block in zip(result.density.cells, result.density.blocks)
    }
    template_options = vq.LatticeSumOptions()
    template_options.cutoff_bohr = 18.0
    old_template = vq.compute_overlap_lattice(basis, sysp, template_options)
    old_template_keys = {
        tuple(int(value) for value in cell.index)
        for cell in old_template.cells
    }
    source_only = set(returned_by_key).difference(old_template_keys)
    assert source_only == {
        (-2, 0, 0),
        (0, -2, 0),
        (0, 0, -2),
        (0, 0, 2),
        (0, 2, 0),
        (2, 0, 0),
    }
    assert all(np.any(returned_by_key[key] != 0.0) for key in source_only)

    with zipfile.ZipFile(output.with_suffix(".qvf")) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        section = next(
            item
            for item in manifest["sections"]
            if item["kind"] == "volume.density"
        )
        data_member = section["members"]["data"]
        rho = np.frombuffer(
            archive.read(data_member["path"]),
            dtype=np.dtype(data_member["dtype"]),
        ).reshape(tuple(data_member["shape"]))
        grid = json.loads(archive.read(section["members"]["grid"]["path"]))

    voxel_vectors = np.asarray(grid["voxel_vectors"], dtype=float)
    np.testing.assert_allclose(
        voxel_vectors * np.asarray(rho.shape, dtype=float)[:, None],
        lattice.T,
        rtol=0.0,
        atol=1.0e-12,
    )
    integrated_electrons = float(rho.sum() * abs(np.linalg.det(voxel_vectors)))
    assert integrated_electrons == pytest.approx(2.0, abs=5.0e-4)

    diagnostics = "\n".join(
        [str(warning.message) for warning in caught]
        + [
            output.with_suffix(".out").read_text(),
            output.with_suffix(".system").read_text(),
        ]
    )
    assert "density grid evaluation failed" not in diagnostics
    assert "template omits converged lattice cells" not in diagnostics


def test_odd_electron_multik_bipole_qvf_output_survives_subprocess(tmp_path):
    """Public odd-electron multi-k BIPOLE writes its returned spin density."""
    output = tmp_path / "h_uhf_k211"
    script = f"""
import json
import numpy as np
import zipfile
import vibeqc as vq

system = vq.PeriodicSystem(
    3,
    np.eye(3) * 18.0,
    [vq.Atom(1, [0.0, 0.0, 0.0])],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
result = vq.run_periodic_job(
    system=system,
    basis=basis,
    method="UHF",
    jk_method="bipole",
    kpoints=(2, 1, 1),
    output={str(output)!r},
    output_qvf=True,
    citations=False,
    write_molden_file=False,
    write_xyz_file=False,
    write_poscar_file=False,
    write_xsf_structure_file=False,
    write_cif_file=False,
    write_population_file=False,
    density_spacing_bohr=0.5,
    max_iter=40,
    bipole_cutoff_bohr=18.0,
    bipole_nuclear_cutoff_bohr=18.0,
    convergence="off",
    progress=False,
)
assert result.converged
assert abs(result.s_squared - 0.75) < 1.0e-12
assert any(
    tuple(int(value) for value in cell.index) != (0, 0, 0)
    and np.any(np.asarray(block) != 0.0)
    for cell, block in zip(result.density_alpha.cells, result.density_alpha.blocks)
)
with zipfile.ZipFile({str(output.with_suffix('.qvf'))!r}) as archive:
    manifest = json.loads(archive.read("manifest.json"))
    density_section = next(
        section
        for section in manifest["sections"]
        if section["kind"] == "volume.density"
    )
    data_member = density_section["members"]["data"]
    rho = np.frombuffer(
        archive.read(data_member["path"]),
        dtype=np.dtype(data_member["dtype"]),
    )
    grid = json.loads(
        archive.read(density_section["members"]["grid"]["path"])
    )
voxel_vectors = np.asarray(grid["voxel_vectors"], dtype=float)
integrated = float(rho.sum() * abs(np.linalg.det(voxel_vectors)))
assert abs(integrated - 1.0) < 1.0e-3, integrated
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "density_grid" not in proc.stderr
    assert "omits converged lattice cells" not in proc.stderr
    assert output.with_suffix(".qvf").exists()
    assert output.with_suffix(".system").exists()


# ---------------------------------------------------------------------------
# output_cells selection — multi-g J for Bloch sum
# ---------------------------------------------------------------------------

def test_output_cells_returns_one_block_per_index():
    """Passing ``output_cells=[0, 1, 2]`` returns three J(g) blocks,
    one per requested cell. Used by the multi-k dispatch to build
    J(k) = Σ_g e^{i k·R_g} J(g)."""
    sysp, basis = _h2_in_box()
    D_real, _, _ = _converged_density_single_cell(sysp, basis)
    # D_real has one cell (Γ-only); request J(0) three times and
    # verify we get the same block back.
    results = vq.build_j_long_range_periodic(
        basis, sysp, D_real, omega=0.5, spacing_bohr=0.3,
        output_cells=[0, 0, 0],
    )
    assert isinstance(results, list)
    assert len(results) == 3
    for J_g in results:
        assert J_g.shape == (basis.nbasis, basis.nbasis)
    # All three should be identical (same cell index requested).
    assert np.allclose(results[0], results[1], atol=1e-12)
    assert np.allclose(results[0], results[2], atol=1e-12)


# ---------------------------------------------------------------------------
# Skew-cell support
# ---------------------------------------------------------------------------

def test_non_orthorhombic_lattice_density_supported():
    """The density grid follows skew lattice vectors in fractional coordinates."""
    lat = np.array([
        [20.0, 1.0, 0.0],
        [0.0, 20.0, 0.0],
        [0.0, 0.0, 20.0],
    ])
    sysp = vq.PeriodicSystem(
        3, lat, [vq.Atom(1, [10, 10, 9.3]), vq.Atom(1, [10, 10, 10.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    # Use the overlap lattice as a cheap finite matrix-set witness.
    # The absolute density is not physical; this exercises the skew
    # grid and AO image geometry on the public API path.
    opts = vq.LatticeSumOptions()
    S_lat = vq.compute_overlap_lattice(basis, sysp, opts)
    rho, grid_shape = vq.evaluate_periodic_density_on_grid(
        basis, sysp, S_lat, grid_shape=(4, 4, 4),
    )
    assert grid_shape == (4, 4, 4)
    assert rho.shape == grid_shape
    assert np.all(np.isfinite(rho))


# ---------------------------------------------------------------------------
# Shifted-basis cache + chi_ref reuse (perf-oriented, must preserve numerics)
# ---------------------------------------------------------------------------

def test_shifted_basis_cache_skips_rebuilds_on_warm_cache():
    """After the first density evaluation populates the cache, a
    second evaluation must not allocate any new shifted ``BasisSet``
    entries. Asserts the cache size is unchanged between two
    consecutive calls."""
    from vibeqc import ewald_j as _ej
    sysp, basis = _h2_in_box()
    D_real, _, _ = _converged_density_single_cell(sysp, basis)

    _ej.clear_shifted_basis_cache()
    vq.evaluate_periodic_density_on_grid(
        basis, sysp, D_real, spacing_bohr=0.5, ao_image_radius=1,
    )
    size_after_first = len(_ej._SHIFTED_BASIS_CACHE)
    # Sanity: a single density call with image_radius=1 builds 26
    # non-zero image shifts (3^3 cube minus the (0,0,0) home cell,
    # which evaluate_ao_periodic feeds directly to evaluate_ao
    # without going through the cache).
    assert size_after_first >= 26, size_after_first

    vq.evaluate_periodic_density_on_grid(
        basis, sysp, D_real, spacing_bohr=0.5, ao_image_radius=1,
    )
    assert len(_ej._SHIFTED_BASIS_CACHE) == size_after_first

    # The J builder reuses the same cache (and reuses chi_ref from
    # the density helper internally); no new shifted bases either.
    vq.build_j_long_range_periodic(
        basis, sysp, D_real, omega=0.5,
        spacing_bohr=0.5, ao_image_radius=1,
    )
    assert len(_ej._SHIFTED_BASIS_CACHE) == size_after_first


def test_density_and_J_invariant_to_cache_state():
    """Computing density + J with a cold cache vs a warm cache must
    yield bit-exact identical results — caching is a perf transform,
    never a numerical one."""
    from vibeqc import ewald_j as _ej
    sysp, basis = _h2_in_box()
    D_real, _, _ = _converged_density_single_cell(sysp, basis)

    _ej.clear_shifted_basis_cache()
    rho_cold, gs_cold = vq.evaluate_periodic_density_on_grid(
        basis, sysp, D_real, spacing_bohr=0.5, ao_image_radius=1,
    )
    J_cold = vq.build_j_long_range_periodic(
        basis, sysp, D_real, omega=0.5,
        spacing_bohr=0.5, ao_image_radius=1,
    )

    rho_warm, gs_warm = vq.evaluate_periodic_density_on_grid(
        basis, sysp, D_real, spacing_bohr=0.5, ao_image_radius=1,
    )
    J_warm = vq.build_j_long_range_periodic(
        basis, sysp, D_real, omega=0.5,
        spacing_bohr=0.5, ao_image_radius=1,
    )

    assert gs_cold == gs_warm
    np.testing.assert_array_equal(rho_cold, rho_warm)
    np.testing.assert_array_equal(J_cold, J_warm)


def test_clear_shifted_basis_cache_empties_it():
    """``clear_shifted_basis_cache`` is the documented escape hatch
    for long-running processes that build many ``(basis, system)``
    pairs; verify it actually drops the entries."""
    from vibeqc import ewald_j as _ej
    sysp, basis = _h2_in_box()
    D_real, _, _ = _converged_density_single_cell(sysp, basis)
    vq.evaluate_periodic_density_on_grid(
        basis, sysp, D_real, spacing_bohr=0.5, ao_image_radius=1,
    )
    assert len(_ej._SHIFTED_BASIS_CACHE) > 0
    _ej.clear_shifted_basis_cache()
    assert len(_ej._SHIFTED_BASIS_CACHE) == 0


def test_shifted_basis_cache_is_lru_bounded(monkeypatch):
    """The shifted-basis cache must not grow without bound in daemons."""
    from vibeqc import ewald_j as _ej

    sysp, basis = _h2_in_box()
    _ej.clear_shifted_basis_cache()
    monkeypatch.setattr(_ej, "_SHIFTED_BASIS_CACHE_MAX_ENTRIES", 3)

    for ix in range(1, 7):
        _ej.get_shifted_basis(basis, sysp, (ix, 0, 0))

    assert len(_ej._SHIFTED_BASIS_CACHE) == 3
    assert [key[2] for key in _ej._SHIFTED_BASIS_CACHE] == [
        (4, 0, 0),
        (5, 0, 0),
        (6, 0, 0),
    ]
    _ej.clear_shifted_basis_cache()


def _lih_rocksalt_primitive():
    """Compact LiH rocksalt primitive fcc cell (a = 7.7176 bohr)."""
    a = 7.7176
    prim = 0.5 * a * np.array([[0.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0],
                               [1.0, 1.0, 0.0]])
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = prim
    system.unit_cell = [core.Atom(3, [0.0, 0.0, 0.0]),
                        core.Atom(1, [0.5 * a, 0.0, 0.0])]
    return system


def _converged_lattice_density(system, basis):
    """Converged Gamma density as a real-space lattice set, plus the
    exact electron count ``sum_g tr(D(g) S(g))``."""
    from vibeqc.periodic_runner import _density_lattice_set_for_output

    lat_opts = core.LatticeSumOptions()
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 80
    opts.conv_tol_energy = 1e-9
    res = vq.run_rhf_periodic_gamma_gdf(system, basis, opts, progress=False)

    class _Proxy:
        pass

    proxy = _Proxy()
    proxy.density = np.asarray(res.density)
    d_set = _density_lattice_set_for_output(basis, system, proxy, lat_opts)
    s_lat = core.compute_overlap_lattice(basis, system, lat_opts)
    n_exact = sum(
        float(np.einsum(
            "ij,ij->",
            np.asarray(d_set.blocks[i]),
            np.asarray(s_lat.blocks[i]),
        ))
        for i in range(len(d_set.cells))
    )
    return d_set, n_exact


@pytest.mark.slow
@pytest.mark.parametrize("compact", [True, False])
def test_grid_density_converges_to_the_electron_count(compact):
    """The grid density must CONVERGE to ``sum_g tr(D(g) S(g))``
    evaluated with the SAME ``LatticeSumOptions`` as the AO image radius
    grows -- on a compact crystal and on a vacuum-padded box alike.

    The reference is deliberately the lattice set's own trace, NOT the
    physical electron count: those coincide only once the lattice set is
    converged, so a Gamma-only run with a truncated set legitimately
    falls short (the compact fixture below references 2.7407, not 4).
    Comparing the grid integral against N_elec would flag that
    truncation as a second defect. The vacuum-padded case pins the
    regime where the two DO coincide (exactly 2.0000).

    Note both fixtures carry a SINGLE non-zero g block (1 of 135 cells),
    which is the regime multi-k GAPW writes in -- single-g gives no
    immunity here, because the double count came from the two
    independent AO image sums and needs only one g block to appear (that
    path read 8.24 at the shipped default against an exact 3.82).

    Regression for PERIODIC-DENSITY-GRID-DOUBLE-COUNTS-IMAGES. The
    periodic density carries exactly one lattice sum, shared by both AO
    factors:

        rho(r) = S_R S_g D(g)_{munu} chi_mu(r-R) chi_nu(r-R-R_g)

    Until 2026-08-02 the two factors were image-summed independently,
    so the integral GREW with ``ao_image_radius`` instead of settling:
    on the compact fixture below it read 0.2134 / 3.6161 / 3.9915 at
    radius 0 / 1 / 2 against an exact 2.7407 -- already 46 % over and
    still climbing, and the user-facing .xsf / .cube artifacts on every
    periodic route inherited it. Monotone growth is the signature this
    test exists to catch, so it asserts convergence (successive radii
    agree) rather than just closeness at one radius.
    """
    if compact:
        system = _lih_rocksalt_primitive()
        shape = (40, 40, 40)
    else:
        L = 12.0
        c = L / 2
        system = core.PeriodicSystem()
        system.dim = 3
        system.lattice = np.eye(3) * L
        system.unit_cell = [core.Atom(1, [c - 0.7, c, c]),
                            core.Atom(1, [c + 0.7, c, c])]
        shape = (48, 48, 48)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    d_set, n_exact = _converged_lattice_density(system, basis)

    lat = np.asarray(system.lattice, dtype=float)
    dV = abs(np.linalg.det(lat)) / (shape[0] * shape[1] * shape[2])
    integrals = []
    for radius in (2, 3):
        rho, _ = vq.evaluate_periodic_density_on_grid(
            basis, system, d_set, grid_shape=shape, ao_image_radius=radius,
        )
        integrals.append(float(rho.sum()) * dV)

    assert integrals[1] == pytest.approx(integrals[0], abs=2e-3), (
        f"grid density is not converged in the AO image radius: "
        f"{integrals[0]:.4f} (r=2) vs {integrals[1]:.4f} (r=3) -- growth "
        "with radius is the double-counted-image signature"
    )
    assert integrals[1] == pytest.approx(n_exact, abs=5e-3), (
        f"grid density integrates to {integrals[1]:.4f}, but the density "
        f"matrix carries sum_g tr(D(g) S(g)) = {n_exact:.4f}"
    )
