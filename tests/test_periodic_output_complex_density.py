"""Periodic output handling for complex multi-k density matrices."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import CoulombMethod, LatticeSumOptions
from vibeqc.periodic_runner import (
    _fold_per_k_density_for_periodic_output,
    _sum_lattice_density_sets_for_output,
)


def _h2_box():
    box = 12.0
    c = box / 2.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [
            vq.Atom(1, [c, c, c - 0.7]),
            vq.Atom(1, [c, c, c + 0.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = LatticeSumOptions()
    opts.coulomb_method = CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 8.0
    return system, basis, opts


def test_per_k_density_fold_materializes_time_reversal_partner_as_real():
    system, basis, lat_opts = _h2_box()
    reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
    k_cart = reciprocal @ np.array([0.25, 0.0, 0.0])
    density_k = np.eye(basis.nbasis, dtype=np.complex128)

    density_set = _fold_per_k_density_for_periodic_output(
        basis,
        system,
        [density_k],
        [k_cart],
        [1.0],
        lat_opts,
    )

    for block in density_set.blocks:
        arr = np.asarray(block)
        assert not np.iscomplexobj(arr)
        assert np.isfinite(arr).all()


def test_per_k_density_fold_symmetrizes_inexact_explicit_pair():
    """Regression: a full +/-k mesh (e.g. Monkhorst-Pack (2,2,2)) must fold
    real even when the per-k lattice-truncation asymmetry of the Fock build
    leaves D(-k) != D(k)^* at the ~1e-5 level. Before the fold was
    time-reversal-symmetrized, c-diamond RHF/STO-3G kpoints=(2,2,2) with
    write_density=True aborted its .density.xsf / .xsf writers with
    "periodic density block g=(0,0,0) has a non-negligible imaginary
    component (max|Im|=4.92e-06)" (qc-input-library artifact
    01706-c-diamond-rhf-sto3g-k222-qvf, v0.15.28)."""
    system, basis, lat_opts = _h2_box()
    reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
    k_cart = reciprocal @ np.array([0.25, 0.0, 0.0])
    n = basis.nbasis
    rng = np.random.default_rng(1706)

    def _hermitian(seed_matrix):
        return 0.5 * (seed_matrix + seed_matrix.conj().T)

    D_plus = _hermitian(
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    )
    # Break exact conjugate pairing by a Hermitian perturbation of the
    # size the per-k truncation asymmetry introduces in practice.
    D_minus = D_plus.conj() + 5.0e-6 * _hermitian(
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    )

    density_set = _fold_per_k_density_for_periodic_output(
        basis,
        system,
        [D_plus, D_minus],
        [k_cart, -k_cart],
        [0.5, 0.5],
        lat_opts,
    )

    for block in density_set.blocks:
        arr = np.asarray(block)
        assert not np.iscomplexobj(arr)
        assert np.isfinite(arr).all()

    # The symmetrized fold is the real part of the pristine fold; at the
    # home cell g=(0,0,0) (phase 1) that is the weighted real average.
    home = next(
        i
        for i, cell in enumerate(density_set.cells)
        if tuple(cell.index) == (0, 0, 0)
    )
    np.testing.assert_allclose(
        np.asarray(density_set.blocks[home]),
        0.5 * D_plus.real + 0.5 * D_minus.real,
        atol=1.0e-12,
    )


def test_per_k_density_fold_symmetrizes_self_inverse_kpoint():
    """A zone-boundary (self-inverse) k point carries a Hermitian but
    complex density; the fold must keep only its time-reversal-symmetric
    (real) part instead of erroring on the imaginary residue."""
    system, basis, lat_opts = _h2_box()
    reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
    k_cart = reciprocal @ np.array([0.5, 0.0, 0.0])
    n = basis.nbasis
    D = np.eye(n, dtype=np.complex128)
    if n > 1:
        D[0, 1] += 1.0e-3j
        D[1, 0] -= 1.0e-3j

    density_set = _fold_per_k_density_for_periodic_output(
        basis,
        system,
        [D],
        [k_cart],
        [1.0],
        lat_opts,
    )

    for block in density_set.blocks:
        arr = np.asarray(block)
        assert not np.iscomplexobj(arr)
        assert np.isfinite(arr).all()


def test_spin_density_set_sum_projects_tiny_imaginary_residual():
    system, basis, lat_opts = _h2_box()
    alpha = _fold_per_k_density_for_periodic_output(
        basis,
        system,
        [np.eye(basis.nbasis, dtype=np.complex128) * (1.0 + 1.0e-12j)],
        [np.zeros(3)],
        [1.0],
        lat_opts,
    )
    beta = _fold_per_k_density_for_periodic_output(
        basis,
        system,
        [np.eye(basis.nbasis, dtype=np.complex128) * (0.5 - 1.0e-12j)],
        [np.zeros(3)],
        [1.0],
        lat_opts,
    )

    total = _sum_lattice_density_sets_for_output(
        basis,
        system,
        lat_opts,
        alpha,
        beta,
        label="periodic spin density",
    )

    for block in total.blocks:
        arr = np.asarray(block)
        assert not np.iscomplexobj(arr)
        assert np.isfinite(arr).all()


def test_spin_density_set_sum_rejects_large_imaginary_residual():
    system, basis, lat_opts = _h2_box()
    cell = SimpleNamespace(index=(0, 0, 0))
    alpha = SimpleNamespace(
        cells=[cell],
        blocks=[np.eye(basis.nbasis, dtype=np.complex128)],
    )
    beta = SimpleNamespace(
        cells=[cell],
        blocks=[np.eye(basis.nbasis, dtype=np.complex128) * (1.0 + 1.0e-3j)],
    )

    with pytest.raises(ValueError, match="periodic spin density.*imaginary"):
        _sum_lattice_density_sets_for_output(
            basis,
            system,
            lat_opts,
            alpha,
            beta,
            label="periodic spin density",
        )


def test_spin_density_set_sum_rejects_misaligned_cell_keys():
    """Equal-sized spin sets still need the same lattice-cell alignment."""
    home = SimpleNamespace(index=(0, 0, 0))
    outer = SimpleNamespace(index=(-1, 0, 0))
    alpha = SimpleNamespace(cells=[home], blocks=[np.eye(1)])
    beta = SimpleNamespace(cells=[outer], blocks=[np.eye(1)])

    with pytest.raises(ValueError, match="different keyed cell order"):
        _sum_lattice_density_sets_for_output(
            None,
            None,
            None,
            alpha,
            beta,
            label="periodic spin density",
        )
