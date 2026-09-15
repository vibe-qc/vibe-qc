# Copyright (c) vibe-qc contributors.
# SPDX-License-Identifier: MPL-2.0
"""Every SCF driver must build its DIIS error from the density that *built* F.

The commutator ``e = F D S - S D F`` vanishes identically whenever ``D`` is
built from the eigenvectors of that same ``F``, for any occupations,
fractional or integer. In the generalized eigenbasis ``F C = S C eps`` with
``C^T S C = I`` and ``D = C f C^T``::

    F D S - S D F = S C (eps f - f eps) C^T S = 0

because ``eps`` and ``f(eps)`` are both diagonal and therefore commute.

So a driver that forms its DIIS error from ``D_out`` (the density it just
diagonalized *out of* F) instead of ``D_in`` (the density that built F) gets
an error vector that is numerically zero on every iteration. Its DIIS then
silently degenerates into a no-op: no extrapolation ever happens, the SCF
still converges on the energy criterion, and every energy-only test passes.

That failure is invisible to the rest of the suite. This file pins it
directly: on a system whose orbitals are *not* fixed by symmetry, the
reported ``grad_norm`` (the commutator norm) must be clearly non-zero on the
early iterations, before convergence has driven it there honestly.

Reported by the scf_acceleration.py audit, 2026-07-10.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2_631g_cell(a: float = 12.0):
    """H2 in a wide cubic cell, 6-31G.

    6-31G, not STO-3G. A minimal-basis H2 cell has one basis function per
    atom, so both MOs are fixed by symmetry, F and D commute exactly, and
    ``grad_norm`` is ~1e-16 on every iteration *for a correct driver too*.
    Such a cell cannot distinguish a healthy commutator from a dead one.
    """
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * a,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])])
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "6-31g")


def _short_run_opts(cls, **kw):
    o = cls()
    if hasattr(o, "lattice_opts"):
        o.lattice_opts.cutoff_bohr = 10.0
        o.lattice_opts.nuclear_cutoff_bohr = 14.0
    o.conv_tol_energy = 1e-14      # unreachable: run the full max_iter
    o.conv_tol_grad = 1e-12
    o.max_iter = 4
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _assert_error_vector_alive(result, driver: str):
    grads = [it.grad_norm for it in result.scf_trace][:3]
    assert max(grads) > 1e-8, (
        f"{driver}: the DIIS error norm is ~0 on every early iteration "
        f"({grads}). The commutator FDS-SDF vanishes identically when D is "
        f"built from the eigenvectors of that same F, so this driver is "
        f"almost certainly forming its error from D_out rather than the "
        f"D_in that built F. Its DIIS is a silent no-op."
    )


def test_cxx_gamma_rhf_error_from_d_in():
    sysp, basis = _h2_631g_cell()
    r = vq.run_rhf_periodic_gamma(
        sysp, basis, _short_run_opts(vq.PeriodicRHFOptions))
    _assert_error_vector_alive(r, "run_rhf_periodic_gamma")


def test_cxx_multi_k_rhf_error_from_d_in():
    sysp, basis = _h2_631g_cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    r = vq.run_rhf_periodic(
        sysp, basis, km, _short_run_opts(vq.PeriodicSCFOptions))
    _assert_error_vector_alive(r, "run_rhf_periodic")


def test_cxx_multi_k_rks_error_from_d_in():
    sysp, basis = _h2_631g_cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    r = vq.run_rks_periodic(
        sysp, basis, km, _short_run_opts(vq.PeriodicKSOptions, functional="pbe"))
    _assert_error_vector_alive(r, "run_rks_periodic")


@pytest.mark.parametrize("driver,opts_cls,kw", [
    ("run_rhf_periodic_gamma_ewald3d", "PeriodicRHFOptions", {}),
    ("run_uhf_periodic_gamma_ewald3d", "PeriodicSCFOptions", {}),
    ("run_rks_periodic_gamma_ewald3d", "PeriodicKSOptions", {"functional": "pbe"}),
    ("run_rhf_periodic_gamma_gdf", "PeriodicRHFOptions", {}),
])
def test_python_gamma_drivers_error_from_d_in(driver, opts_cls, kw):
    sysp, basis = _h2_631g_cell()
    run = getattr(vq, driver)
    opts = _short_run_opts(getattr(vq, opts_cls), **kw)
    _assert_error_vector_alive(run(sysp, basis, opts), driver)


def test_python_multi_k_ewald_rhf_error_from_d_in():
    sysp, basis = _h2_631g_cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _short_run_opts(vq.PeriodicSCFOptions))
    _assert_error_vector_alive(r, "run_rhf_periodic_multi_k_ewald3d")


def test_commutator_vanishes_on_d_out_even_with_fractional_occupations():
    """The algebra behind this whole file, pinned directly.

    Also refutes the claim that the commutator error degrades under
    smearing: with Fermi-Dirac fractional occupations the commutator at a
    self-consistent D is just as zero as with integer ones, because eps and
    f(eps) are both diagonal in the same basis. A density-residual error
    vector is not needed to make DIIS well-defined at T > 0.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(1)
    n = 6
    A = rng.normal(size=(n, n))
    S = A @ A.T + n * np.eye(n)
    B = rng.normal(size=(n, n))
    F = B + B.T
    eps, C = scipy_linalg.eigh(F, S)

    integer_occ = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
    fermi_occ = 2.0 / (1.0 + np.exp((eps - eps[2]) / 0.35))

    for label, f in (("integer", integer_occ), ("fermi-dirac", fermi_occ)):
        D = (C * f) @ C.T                      # D built from F's own vectors
        commutator = F @ D @ S - S @ D @ F
        assert np.linalg.norm(commutator) < 1e-10, (
            f"{label}: commutator should vanish identically at D_out"
        )
    # And the fractional occupations really are fractional.
    assert np.any((fermi_occ > 1e-3) & (fermi_occ < 2.0 - 1e-3))
