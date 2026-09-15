"""The short-range aux ERI kernels must enumerate pair-complete cells.

`pair_complete_cells` (cpp/src/aux_eri.cpp) exists because bounding the
lattice sum by |g| <= cutoff is not translation invariant: the
translations that matter form a ball centred on the intra-cell offset
O_P - O_Q, not on the origin. The 2026-08-03 fix `5a40626a8` converted
every bare-Coulomb kernel and MISSED the two short-range ones, which kept
the old `direct_lattice_cells` enumeration until 2026-08-06.

Measured on MgO primitive FCC / def2-svp-jk at omega=0.4, cutoff 12 bohr,
comparing the cell against the SAME crystal with one atom translated by a
lattice vector:

    before:  min eig(M_SR) -2.14e+01 vs -3.30e+01,  ||dM|| = 1.27e+02
    after :  identical spectra,                     ||dM|| = 1.66e-13

A single-atom cell has zero intra-cell offset, so vacuum-box fixtures
cannot see this.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq


@pytest.mark.parametrize("maximum_order", [0, 6, 24])
@pytest.mark.parametrize("argument", [0., 1., np.nextafter(117., 0.), 117.,
                                     np.nextafter(117., np.inf), 10000.])
def test_libint_boys_table_endpoint_matches_incomplete_gamma(argument, maximum_order):
    """#754: T=117 used to read past the final Chebyshev interval."""
    from scipy.special import gammainc, gammaln
    from vibeqc import _vibeqc_core as core

    order = np.arange(maximum_order + 1)
    if argument == 0:
        reference = 1. / (2*order + 1)
    else:
        a = order + .5
        reference = np.exp(gammaln(a) - np.log(2.) - a*np.log(argument)) * gammainc(a, argument)
    np.testing.assert_allclose(
        core._libint_boys_values(argument, maximum_order), reference, rtol=2e-13, atol=0,
    )
from vibeqc._vibeqc_core import (
    compute_2c_eri_lattice_sr,
    compute_3c_eri_lattice,
    compute_3c_eri_lattice_sr,
)
from vibeqc.aux_basis import make_aux_basis_set

BOHR = 0.529177210903
OMEGA = 0.4


def _small_sr_basis(shift=False):
    """Unnormalized s Gaussians allow an independent Boys-function oracle."""
    core = vq._vibeqc_core
    lattice = np.array([[3.8, 0.4, 0.1], [0.0, 4.1, 0.3], [0.0, 0.0, 4.3]])
    centers = [np.array([0.2, 0.3, 0.4]), np.array([1.5, 0.7, 0.8])]
    if shift:
        centers[1] = centers[1] + lattice[:, 0]
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, r) for r in centers])
    mol = system.unit_cell_molecule()
    def basis(exponents):
        return core.BasisSet(mol, [
            core.ShellInfo(i, 0, True, [a], [1.0], centers[i])
            for i, a in enumerate(exponents)
        ], "sr-oracle", coefficients_pre_normalized=True)
    return system, basis([0.7, 1.1]), basis([0.6, 0.9])


def _sr_three_center_oracle(system, basis, aux, q, k, omega, cutoff):
    """Gaussian product theorem + analytic ssss Boys integrals, no DF code.

    With p=a+b, rho=c*p/(c+p), the SR factor is
    F0(rho*R^2) - eta*F0(eta^2*rho*R^2), eta=omega/sqrt(omega^2+rho).
    This independently evaluates the double-image sum in Ye 2021 Eq. (12)
    (before its finite zero-mode subtraction).
    """
    from itertools import product
    from scipy.special import hyp1f1
    lattice = np.asarray(system.lattice)
    translations = np.asarray(list(product(range(-5, 6), repeat=3))) @ lattice.T
    out = np.zeros((aux.nbasis, basis.nbasis, basis.nbasis), complex)
    for ip, sp in enumerate(aux.shells()):
        c = sp.exponents[0]
        P = np.asarray(sp.origin) + translations
        for im, sm in enumerate(basis.shells()):
            a, A = sm.exponents[0], np.asarray(sm.origin)
            for inn, sn in enumerate(basis.shells()):
                b, B0 = sn.exponents[0], np.asarray(sn.origin)
                p = a + b
                rho = c * p / (c + p)
                eta = omega / np.sqrt(omega * omega + rho)
                pref = 2 * np.pi**2.5 / (c * p * np.sqrt(c + p))
                for R in translations:
                    B = B0 + R
                    segment = B - A
                    d2 = segment @ segment
                    if d2 > cutoff**2:
                        continue
                    t = np.clip((P - A) @ segment / d2, 0, 1) if d2 else np.zeros(len(P))
                    keep = np.sum((P - A - t[:, None] * segment)**2, axis=1) <= cutoff**2
                    X = (a * A + b * B) / p
                    u = rho * np.sum((P[keep] - X)**2, axis=1)
                    integral = pref * np.exp(-a*b/p*d2) * (
                        hyp1f1(0.5, 1.5, -u) - eta * hyp1f1(0.5, 1.5, -eta*eta*u)
                    )
                    out[ip, im, inn] += np.exp(1j * k @ R) * np.sum(
                        np.exp(-1j * (translations[keep] @ q)) * integral
                    )
    return out


def test_double_image_sr_matches_independent_boys_oracle():
    from vibeqc._vibeqc_core import compute_gdf_sr_three_center
    system, basis, aux = _small_sr_basis()
    q = np.array([0.17, -0.09, 0.11])
    ks = np.array([[0.0, 0.0, 0.0], [0.23, 0.13, -0.07]])
    actual = compute_gdf_sr_three_center(
        basis, aux, system, q, ks, 0.65, 5.0, 5.0, 4096, 100000
    )
    for ik, k in enumerate(ks):
        expected = _sr_three_center_oracle(system, basis, aux, q, k, 0.65, 5.0)
        np.testing.assert_allclose(actual[ik], expected, atol=2e-12, rtol=2e-12)


def test_double_image_sr_atom_translation_covariance():
    from vibeqc._vibeqc_core import compute_gdf_sr_three_center, compute_gdf_sr_metric
    q = np.array([0.17, -0.09, 0.11])
    k = np.array([0.23, 0.13, -0.07])
    arrays = []
    for shift in (False, True):
        system, basis, aux = _small_sr_basis(shift)
        arrays.append((
            compute_gdf_sr_metric(aux, system, q, 0.65, 5.0, 4096, 100000),
            compute_gdf_sr_three_center(basis, aux, system, q, k[None], 0.65, 5.0, 5.0, 4096, 100000)[0],
        ))
    L = np.asarray(system.lattice)[:, 0]
    aux_phase = np.exp(1j * np.array([0, q @ L]))
    bra_phase = np.exp(1j * np.array([0, (k-q) @ L]))
    ket_phase = np.exp(-1j * np.array([0, k @ L]))
    np.testing.assert_allclose(
        arrays[1][0], arrays[0][0] * aux_phase[:, None] * aux_phase.conj()[None],
        atol=2e-12,
    )
    np.testing.assert_allclose(
        arrays[1][1], arrays[0][1] * aux_phase[:, None, None]
        * bra_phase[None, :, None] * ket_phase[None, None, :], atol=2e-12,
    )


def test_double_image_sr_limits_before_allocation():
    from vibeqc._vibeqc_core import compute_gdf_sr_three_center, compute_gdf_sr_metric
    system, basis, aux = _small_sr_basis()
    with pytest.raises((ValueError, RuntimeError), match="output byte cap"):
        compute_gdf_sr_three_center(
            basis, aux, system, np.zeros(3), np.zeros((1, 3)),
            0.65, 5.0, 5.0, 1, 100000,
        )
    with pytest.raises((ValueError, RuntimeError), match="image candidate cap"):
        compute_gdf_sr_metric(aux, system, np.zeros(3), 0.65, 1e8, 4096, 1000)
    with pytest.raises(ValueError, match="engine workspace byte cap"):
        compute_gdf_sr_metric(
            aux, system, np.zeros(3), 0.65, 5.0, 4096, 100000,
            workspace_byte_cap=1,
        )


def test_combined_source_admits_metric_tensor_and_fourier_workspace_together():
    from vibeqc._vibeqc_core import compute_gdf_range_separated_integrals
    system, basis, aux = _small_sr_basis()
    args = (basis, aux, system, np.zeros(3), np.zeros((2, 3)),
            np.zeros((1, 3)), 0.65, 5.0, 5.0)
    output_bytes = 16 * (aux.nbasis**2 + 2*aux.nbasis*basis.nbasis**2)
    with pytest.raises(ValueError, match="combined output byte cap"):
        compute_gdf_range_separated_integrals(*args, output_bytes-1, 2**20, 100000)
    with pytest.raises(ValueError, match="Fourier transient byte cap"):
        compute_gdf_range_separated_integrals(*args, output_bytes, 1, 100000)
    M, T, ng = compute_gdf_range_separated_integrals(*args, output_bytes, 2**20, 100000)
    assert M.nbytes + T.nbytes == output_bytes
    assert ng == 0  # This deliberately incomplete mesh contains only p=0.
    assert not M.flags.owndata and not T.flags.owndata  # native capsule owners
    empty_metric, tensor_only, ng_only = compute_gdf_range_separated_integrals(
        *args, T.nbytes, 2**20, 100000, compute_metric=False,
    )
    assert empty_metric.shape == (0, 0) and ng_only == ng
    np.testing.assert_array_equal(tensor_only, T)


def test_range_separated_thread_and_workspace_caps_preserve_integrals():
    from vibeqc.aux_basis import _rsgdf_shifted_dense_g_mesh
    core = vq._vibeqc_core
    system, basis, aux = _small_sr_basis()
    q = np.array([0.17, -0.09, 0.11])
    ks = np.arange(24, dtype=float).reshape(8, 3) / 100
    vectors = _rsgdf_shifted_dense_g_mesh(system, q, 30.0)
    shared = 16*aux.nbasis*(128+1) + 8*128
    per_worker = 16*16*128 + core._PERIODIC_AOPAIR_FOURIER_FIXED_NUMERIC_WORKSPACE_BYTES
    original_threads = core.get_num_threads()
    results = []
    try:
        for threads, slots in ((1, 8), (8, 4), (8, 8)):
            core.set_num_threads(threads)
            M, T, _ = core.compute_gdf_range_separated_integrals(
                basis, aux, system, q, ks, vectors, 0.65, 5.0, 5.0,
                4096, shared + slots*per_worker, 100000,
            )
            results.append((M, T))
    finally:
        core.set_num_threads(original_threads)
    for M, T in results[1:]:
        np.testing.assert_array_equal(M, results[0][0])
        np.testing.assert_array_equal(T, results[0][1])


def test_reduced_center_engine_avoids_four_center_primitive_storage():
    """The old construct-then-set idiom retains nprim^4 data in libint.

    Twelve primitives and one worker would consume about 170 MB for the
    prototype plus worker in that idiom. A two-center engine needs only
    nprim^2 data; measure a fresh child's peak growth to catch that difference.
    """
    import json
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import json, resource, sys
        import numpy as np
        import vibeqc as vq
        from vibeqc import _vibeqc_core as core
        system = vq.PeriodicSystem(3, np.eye(3)*4, [vq.Atom(2, [0, 0, 0])])
        basis = core.BasisSet(system.unit_cell_molecule(), [
            core.ShellInfo(0, 0, True, np.linspace(0.7, 1.9, 12),
                           np.full(12, 0.1), [0, 0, 0])
        ], "engine-memory", coefficients_pre_normalized=True)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metric = core.compute_gdf_sr_metric(
            basis, system, np.zeros(3), 0.65, 5.0, 4096, 100000,
            workspace_byte_cap=2**21)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if sys.platform == "darwin" else 1024
        print(json.dumps({"growth_bytes": (after-before)*scale,
                          "finite": bool(np.isfinite(metric).all())}))
    """)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1")
    child = subprocess.run(
        [sys.executable, "-c", script], env=env, text=True,
        capture_output=True, check=True, timeout=60,
    )
    measured = json.loads(child.stdout)
    assert measured["finite"]
    assert measured["growth_bytes"] < 32 * 2**20


@pytest.mark.parametrize("qfrac", [(0.0, 0.0, 0.0), (0.23, -0.17, 0.11)])
def test_range_separated_integrals_are_split_parameter_invariant(qfrac):
    from vibeqc._vibeqc_core import compute_gdf_range_separated_integrals
    from vibeqc.aux_basis import _rsgdf_shifted_dense_g_mesh
    system, basis, aux = _small_sr_basis()
    q = np.asarray(system.reciprocal_lattice()) @ qfrac
    ks = np.array([[0.0, 0.0, 0.0], [0.23, 0.13, -0.07]])
    results = []
    for omega in (0.45, 0.8):
        # exp(-p^2/(4*omega^2)) <= 1e-14 outside this sphere.
        cutoff = 2 * omega**2 * np.log(1e14)
        vectors = _rsgdf_shifted_dense_g_mesh(system, q, cutoff)
        M, T, ng = compute_gdf_range_separated_integrals(
            basis, aux, system, q, ks, vectors, omega, 12.0, 18.0,
            4096, 2**20, 1000000,
        )
        assert ng < 2000
        np.testing.assert_allclose(M, M.conj().T, atol=2e-11)
        assert np.linalg.eigvalsh(M).min() > 0
        results.append((M, T))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("qfrac", [(0.0, 0.0, 0.0), (0.23, -0.17, 0.11)])
def test_range_separated_integrals_match_analytic_reciprocal_sum(qfrac):
    """Absolute jellium convention, independent of the native Fourier code."""
    from itertools import product
    from vibeqc._vibeqc_core import compute_gdf_range_separated_integrals
    from vibeqc.aux_basis import _rsgdf_shifted_dense_g_mesh

    system, basis, aux = _small_sr_basis()
    lattice = np.asarray(system.lattice)
    q = np.asarray(system.reciprocal_lattice()) @ qfrac
    k = np.array([0.23, 0.13, -0.07])
    vectors = _rsgdf_shifted_dense_g_mesh(system, q, 80.0)
    g2 = np.sum(vectors**2, axis=1)
    vectors, g2 = vectors[g2 > 1e-20], g2[g2 > 1e-20]
    weights = 4 * np.pi / abs(np.linalg.det(lattice)) / g2
    aft = np.asarray([
        (np.pi / s.exponents[0])**1.5
        * np.exp(-g2 / (4 * s.exponents[0]) - 1j * (vectors @ s.origin))
        for s in aux.shells()
    ])
    pairft = np.zeros((basis.nbasis, basis.nbasis, len(vectors)), complex)
    translations = np.asarray(list(product(range(-5, 6), repeat=3))) @ lattice.T
    for mu, sm in enumerate(basis.shells()):
        a, A = sm.exponents[0], np.asarray(sm.origin)
        for nu, sn in enumerate(basis.shells()):
            b, B0 = sn.exponents[0], np.asarray(sn.origin)
            p = a + b
            for R in translations:
                B = B0 + R
                d2 = np.sum((A-B)**2)
                if d2 > 12.0**2:
                    continue
                X = (a*A + b*B) / p
                pairft[mu, nu] += (np.pi/p)**1.5 * np.exp(
                    -a*b/p*d2 - g2/(4*p) - 1j*(vectors @ X) + 1j*(k @ R)
                )
    expected_m = (aft.conj() * weights) @ aft.T
    expected_t = np.einsum("pg,mng,g->pmn", aft.conj(), pairft, weights)
    results = []
    omega = 0.45
    compact = _rsgdf_shifted_dense_g_mesh(system, q, 2*omega**2*np.log(1e14))
    for auxiliary_cutoff in (12.0, 18.0):
        M, T, _ = compute_gdf_range_separated_integrals(
            basis, aux, system, q, k[None], compact, omega,
            12.0, auxiliary_cutoff, 4096, 2**20, 1000000,
        )
        results.append((M, T[0]))
    error = [np.linalg.norm(M-expected_m) + np.linalg.norm(T-expected_t) for M, T in results]
    assert error[1] < error[0] * 1e-3
    np.testing.assert_allclose(results[1][0], expected_m, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(results[1][1], expected_t, atol=2e-11, rtol=2e-11)


def test_range_separated_pure_angular_shells_and_reciprocal_relabelling():
    """SR and LR must share libint's d/f-shell normalization and Bloch gauge."""
    from vibeqc._vibeqc_core import compute_gdf_range_separated_integrals
    from vibeqc.aux_basis import _rsgdf_shifted_dense_g_mesh
    core = vq._vibeqc_core
    system, _, _ = _small_sr_basis()
    mol = system.unit_cell_molecule()
    basis = core.BasisSet(mol, [
        core.ShellInfo(0, 2, True, [0.8], [1.0], [0.2, 0.3, 0.4]),
    ], "sr-d")
    aux = core.BasisSet(mol, [
        core.ShellInfo(1, 1, True, [0.9], [1.0], [1.5, 0.7, 0.8]),
        core.ShellInfo(0, 3, True, [0.8], [1.0], [0.2, 0.3, 0.4]),
    ], "sr-pf")
    reciprocal = np.asarray(system.reciprocal_lattice())
    q = reciprocal @ [0.23, -0.17, 0.11]
    k = np.array([0.23, 0.13, -0.07])
    results = []
    for omega, shift in ((0.45, np.zeros(3)), (0.8, reciprocal[:, 0])):
        q_shift = q + shift
        compact = _rsgdf_shifted_dense_g_mesh(system, q_shift, 2*omega**2*np.log(1e14))
        M, T, _ = compute_gdf_range_separated_integrals(
            basis, aux, system, q_shift, (k + shift)[None], compact, omega,
            12.0, 18.0, 2**20, 2**20, 1000000,
        )
        results.append((M, T))
    np.testing.assert_allclose(results[0][0], results[1][0], atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-10, rtol=2e-10)


def test_tight_core_integrals_use_an_exponent_independent_reciprocal_mesh():
    """Cubic jellium Gaussian self integrals have an analytic tight limit.

    The regular periodic potential is -xi + 2*pi*r^2/(3*volume).
    Its convolution adds the Gaussian second-moment term below; cubic
    harmonic terms average to zero for these spherical densities.
    """
    from vibeqc._vibeqc_core import compute_gdf_range_separated_integrals
    from vibeqc.aux_basis import _rsgdf_shifted_dense_g_mesh
    core = vq._vibeqc_core
    side, omega = 4.0, 0.6
    volume = side**3
    system = vq.PeriodicSystem(3, np.eye(3)*side, [vq.Atom(2, [0, 0, 0])])
    compact = _rsgdf_shifted_dense_g_mesh(system, np.zeros(3), 2*omega**2*np.log(1e14))
    xi = 2.837297479480619 / side
    counts = []
    for exponent in (20.0, 50000.0, 160000.0):
        normalization = (2*exponent/np.pi)**0.75
        basis = core.BasisSet(system.unit_cell_molecule(), [
            core.ShellInfo(0, 0, True, [exponent], [normalization], [0, 0, 0]),
        ], "tight-s", coefficients_pre_normalized=True)
        M, T, ng = compute_gdf_range_separated_integrals(
            basis, basis, system, np.zeros(3), np.zeros((1, 3)), compact,
            omega, 12.0, 14.0, 4096, 2**20, 1000000,
        )
        charge = (2*exponent/np.pi)**0.75 * (np.pi/exponent)**1.5
        self_metric = 4*np.pi/exponent
        self_three_center = 2**2.25 * np.pi**0.25 / np.sqrt(3) / exponent**0.25
        expected_m = self_metric + charge**2 * (-xi + 2*np.pi/(volume*exponent))
        expected_t = self_three_center + charge * (-xi + 1.5*np.pi/(volume*exponent))
        np.testing.assert_allclose(M[0, 0], expected_m, atol=2e-12, rtol=2e-11)
        np.testing.assert_allclose(T[0, 0, 0, 0], expected_t, atol=2e-12, rtol=2e-11)
        counts.append(ng)
    assert len(set(counts)) == 1
    assert counts[0] < 2000


def _mgo(shift_frac=(0, 0, 0)):
    """MgO primitive FCC; `shift_frac` translates O by lattice vectors.

    Translating an atom by a whole lattice vector describes the IDENTICAL
    crystal, so every physical quantity must be invariant under it.
    """
    a = 4.212 / BOHR
    lat = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]).T
    off = lat @ np.asarray(shift_frac, dtype=float)
    return vq.PeriodicSystem(
        3, lat,
        [vq.Atom(12, [0.0, 0.0, 0.0]),
         vq.Atom(8, list(lat @ np.array([0.5, 0.5, 0.5]) + off))],
    )


def _opts(cutoff):
    o = vq.LatticeSumOptions()
    o.cutoff_bohr = cutoff
    return o


def test_sr_2c_metric_is_translation_invariant():
    """M_SR must not change when an atom moves by a lattice vector."""
    mats = []
    for shift in ((0, 0, 0), (1, 0, 0), (0, -1, 1)):
        sysp = _mgo(shift)
        aux = make_aux_basis_set(sysp.unit_cell_molecule(),
                                 aux_name="def2-svp-jk")
        mats.append(np.asarray(
            compute_2c_eri_lattice_sr(aux, sysp, _opts(12.0), OMEGA)))

    for i, m in enumerate(mats[1:], start=1):
        drift = float(np.linalg.norm(m - mats[0]))
        assert drift < 1e-10, (
            f"shift {i}: M_SR moved by {drift:.3e} under a pure lattice "
            f"translation of one atom, which describes the same crystal. "
            f"The SR kernel is enumerating cells by |g| instead of by pair "
            f"separation (pair_complete_cells)."
        )


def test_sr_3c_tensor_approaches_the_bare_kernel_as_omega_vanishes():
    """T_SR must converge to the bare-Coulomb T as omega -> 0.

    NOT a translation-invariance test. T_Pmn = sum_g (P_0 | mu_0 nu_g)
    pins BOTH the auxiliary index and mu to the reference cell and sums
    only over nu, so translating an atom by a lattice vector genuinely
    changes it -- the bare kernel drifts by 8.6 on this cell and the SR
    one by 1.1, and neither is a defect. Only the 2c metric, where both
    indices are summed, must be invariant (see the test above).

    What DOES pin the SR 3c kernel is its limit: erfc(omega r)/r -> 1/r as
    omega -> 0, so T_SR must approach the bare tensor. Measured on MgO /
    sto-3g / def2-svp-jk at cutoff 20 bohr, relative deviation
    0.49 / 0.16 / 0.032 / 0.0065 at omega 0.2 / 0.05 / 0.01 / 0.002 --
    linear in omega, as erfc(omega r)/r ~ 1/r - 2 omega/sqrt(pi) predicts.
    """
    sysp = _mgo()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    aux = make_aux_basis_set(sysp.unit_cell_molecule(),
                             aux_name="def2-svp-jk")
    opts = _opts(20.0)

    bare = np.asarray(compute_3c_eri_lattice(basis, aux, sysp, opts))
    scale = float(np.linalg.norm(bare))

    devs = []
    for omega in (0.05, 0.01, 0.002):
        sr = np.asarray(
            compute_3c_eri_lattice_sr(basis, aux, sysp, opts, omega))
        devs.append(float(np.linalg.norm(sr - bare)) / scale)

    assert devs[0] > devs[1] > devs[2], (
        f"T_SR must approach the bare tensor monotonically as omega -> 0; "
        f"got {devs}"
    )
    assert devs[-1] < 0.02, (
        f"at omega=0.002 the SR kernel should be within ~1% of bare; got "
        f"{devs[-1]:.3e}"
    )


@pytest.mark.slow
def test_rsgdf_metric_reaches_psd_at_a_converged_cutoff():
    """M_SR + M_LR is a Gram matrix, so it must be PSD once converged.

    It is an inner product in a positive-definite kernel -- the erfc
    kernel's Fourier transform is 4 pi/G^2 (1 - exp(-G^2/4 omega^2)) >= 0 --
    so any negative eigenvalue is truncation, not physics. Measured here
    (MgO, def2-svp-jk, omega=0.4): -16.4 at 8 bohr, -0.78 at 12, -8.2e-05
    at 20, and -3.5e-13 at 30. The LR mesh is already converged at 169
    G-points, so the SR real-space cutoff is the controlling parameter.

    This pins the direction, not one number: the metric must approach PSD
    as the cutoff grows, and must actually get there.
    """
    from vibeqc.aux_basis import rsgdf_g_mesh, rsgdf_lr_2c_metric

    sysp = _mgo()
    aux = make_aux_basis_set(sysp.unit_cell_molecule(),
                             aux_name="def2-svp-jk")
    g_mesh = rsgdf_g_mesh(sysp, OMEGA)
    lr = np.asarray(rsgdf_lr_2c_metric(aux, sysp, OMEGA, g_mesh=g_mesh))

    mins = []
    for cutoff in (12.0, 20.0, 30.0):
        m = np.asarray(
            compute_2c_eri_lattice_sr(aux, sysp, _opts(cutoff), OMEGA)) + lr
        mins.append(float(np.linalg.eigvalsh(0.5 * (m + m.T)).min()))

    assert mins[0] < mins[1] < mins[2], (
        f"min eig must rise monotonically toward 0 as the SR sum "
        f"converges; got {mins}"
    )
    assert mins[-1] > -1e-9, (
        f"the converged metric must be positive semi-definite; got "
        f"{mins[-1]:.3e} at 30 bohr"
    )


@pytest.mark.parametrize('angular', [0, 1, 2, 3])
@pytest.mark.parametrize('omega', [0.4, 1.1])
def test_sr_shell_envelope_screen_bounds_accumulated_raw_integral_error(angular, omega):
    """Angular, signed-contracted shells must obey the requested whole-sum bound."""
    core = vq._vibeqc_core
    system, _, _ = _small_sr_basis()
    mol = system.unit_cell_molecule()
    basis = core.BasisSet(mol, [
        core.ShellInfo(0, angular, True, [2.7, .9], [.7, -.2], [.2, .3, .4]),
        core.ShellInfo(1, 0, True, [1.2], [1.], [1.5, .7, .8]),
    ], 'screen-angular')
    aux = core.BasisSet(mol, [
        core.ShellInfo(0, angular, True, [1.8, .6], [.8, -.3], [.2, .3, .4]),
    ], 'screen-aux')
    q = np.asarray(system.reciprocal_lattice()) @ [.21, -.17, .07]
    k = np.array([[.1, -.2, .3], [-.15, .03, .21]])
    common = (2**24, 1000000, 2**24)
    reference_m = core.compute_gdf_sr_metric(aux, system, q, omega, 10., *common)
    reference_t = core.compute_gdf_sr_three_center(basis, aux, system, q, k, omega, 8., 10., *common)
    differences = []
    for error in (1e-3, 1e-8, 1e-12):
        actual_m = core.compute_gdf_sr_metric(aux, system, q, omega, 10., *common, integral_screen_error=error)
        actual_t = core.compute_gdf_sr_three_center(basis, aux, system, q, k, omega, 8., 10., *common, integral_screen_error=error)
        dm = np.max(np.abs(actual_m-reference_m))
        dt = np.max(np.abs(actual_t-reference_t))
        assert max(dm, dt) <= error + 5e-13
        differences.append(max(dm, dt))
    assert differences[-1] <= differences[0] + 5e-13
    assert differences[0] > 1e-14  # exercise actual skipped contributions


def test_sr_envelope_screen_preserves_dense_core_compact_mesh_oracle():
    core = vq._vibeqc_core
    from vibeqc.aux_basis import _rsgdf_shifted_dense_g_mesh
    system = vq.PeriodicSystem(3, np.eye(3)*4., [vq.Atom(2, [0., 0., 0.])])
    vectors = _rsgdf_shifted_dense_g_mesh(system, np.zeros(3), 20.)
    for exponent in (20., 50000., 160000.):
        basis = core.BasisSet(system.unit_cell_molecule(), [
            core.ShellInfo(0, 0, True, [exponent], [1.], [0., 0., 0.]),
        ], 'screen-tight')
        args = (basis, basis, system, np.zeros(3), np.zeros((1,3)), vectors,
                .6, 12., 14., 4096, 2**20, 1000000)
        reference = core.compute_gdf_range_separated_integrals(*args)
        screened = core.compute_gdf_range_separated_integrals(*args, integral_screen_error=1e-12)
        np.testing.assert_allclose(screened[0], reference[0], rtol=0, atol=2e-12)
        np.testing.assert_allclose(screened[1], reference[1], rtol=0, atol=2e-12)
        assert screened[2] == reference[2] < 2000


def test_sr_primitive_blocks_preserve_signed_contracted_normalization():
    """Long contractions equal an independent sum over separate primitive AOs."""
    core = vq._vibeqc_core
    system = vq.PeriodicSystem(
        3, np.eye(3) * 20.,
        [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [.3, .1, .2])],
    )
    mol = system.unit_cell_molecule()

    def basis(n, angular, center, split):
        exponents = np.geomspace(.6, 80., n)
        coefficients = np.cos(np.arange(n)) * (2 * exponents / np.pi)**.75 / n
        atom_index = 0 if center == [0., 0., 0.] else 1
        shells = ([core.ShellInfo(atom_index, angular, True, [a], [c], center)
                   for a, c in zip(exponents, coefficients)] if split else
                  [core.ShellInfo(atom_index, angular, True, list(exponents), list(coefficients), center)])
        return core.BasisSet(mol, shells, 'primitive-block-check', coefficients_pre_normalized=True)
    orbital = basis(9, 1, [0., 0., 0.], False)
    aux = basis(13, 0, [.3, .1, .2], False)
    primitive_orbital = basis(9, 1, [0., 0., 0.], True)
    primitive_aux = basis(13, 0, [.3, .1, .2], True)
    q, k = np.array([.1, -.2, .3]), np.array([[.02, -.15, .1]])
    # A small workspace admits bounded engines even for a 13-primitive shell.
    caps = (2**24, 10000, 2**20)
    m = core.compute_gdf_sr_metric(aux, system, q, .6, 2., *caps)
    mp = core.compute_gdf_sr_metric(primitive_aux, system, q, .6, 2., *caps)
    t = core.compute_gdf_sr_three_center(orbital, aux, system, q, k, .6, 2., 2., *caps)
    tp = core.compute_gdf_sr_three_center(primitive_orbital, primitive_aux, system, q, k, .6, 2., 2., *caps)
    np.testing.assert_allclose(m[0,0], mp.sum(), rtol=0, atol=2e-13)
    expected = tp.reshape(1, 13, 9, 3, 9, 3).sum(axis=(1, 2, 4))
    np.testing.assert_allclose(t[:,0], expected, rtol=0, atol=2e-13)


def _weighted_gradient_fixture(
    angular, displacement=None, *, orbital_angular=1, image_shift=False,
):
    core = vq._vibeqc_core
    lattice = np.array([[8., .7, .2], [0., 7.8, .5], [0., 0., 8.2]])
    centers = np.array([[.2, .3, .4], [1.4, .7, 1.2]])
    positions = centers.copy()
    if image_shift:
        positions[1] += lattice[:, 0]
    if displacement is not None:
        atom, axis, step = displacement
        positions[atom, axis] += step
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, r) for r in positions])
    molecule = system.unit_cell_molecule()
    orbital = core.BasisSet(molecule, [
        core.ShellInfo(0, orbital_angular, True, [.9, .3], [1., -.15], positions[0]),
        core.ShellInfo(1, 0, True, [.6, .2], [1., .2], positions[1]),
    ], 'sr-gradient-orbital')
    auxiliary = core.BasisSet(molecule, [
        core.ShellInfo(atom, angular, True, [.35, .6, 1.2, 3., 8.],
                       [1., -.1, .2, -.05, .01], positions[atom])
        for atom in range(2)
    ], 'sr-gradient-auxiliary')
    return system, orbital, auxiliary


@pytest.mark.parametrize('angular', [0, 1, 2, 3])
def test_modrho_tensor_scaling_matches_rebuilt_integrals_and_moments(angular):
    from scipy.special import gamma
    from vibeqc.aux_basis import make_modrho_aux_basis, modrho_scales

    core = vq._vibeqc_core
    system, orbital, auxiliary = _weighted_gradient_fixture(angular)
    rebuilt = make_modrho_aux_basis(auxiliary, system.unit_cell_molecule())
    scales = modrho_scales(auxiliary)
    q = np.array([.13, -.09, .07])
    kpoints = np.array([[.2, -.1, .3], [-.15, .25, -.1]])
    caps = (2**20, 100000, 32 * 2**20)
    matrices, tensors = [], []
    for aux in (auxiliary, rebuilt):
        matrices.append(core.compute_gdf_sr_metric(aux, system, q, .6, 5.1, *caps))
        tensors.append(core.compute_gdf_sr_three_center(
            orbital, aux, system, q, kpoints, .6, 4.3, 5.1, *caps,
        ))
    np.testing.assert_allclose(matrices[1], scales[:, None] * matrices[0] * scales[None, :],
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tensors[1], scales[None, :, None, None] * tensors[0],
                               rtol=2e-12, atol=2e-12)
    # Integral of chi_Lm against r^L Y_Lm, where integral Y_Lm^2 dOmega=1.
    # Libint's solid harmonic contributes sqrt(4pi/(2L+1)). This checks
    # the physical moment independently of either rescaling helper.
    for shell in rebuilt.shells():
        alpha = np.asarray(shell.exponents)
        radial = .5 * gamma(angular + 1.5) / alpha**(angular + 1.5)
        moment = np.sqrt(4 * np.pi / (2 * angular + 1)) * np.dot(shell.coefficients, radial)
        assert moment == pytest.approx(1 / np.sqrt(4 * np.pi), rel=2e-14)


@pytest.mark.parametrize('angular,screen,gamma', [
    (0, 0., False), (1, 0., True), (2, 1e-10, False), (3, 0., False),
])
def test_weighted_sr_derivatives_match_raw_source_finite_differences(angular, screen, gamma):
    core = vq._vibeqc_core
    def fixture(displacement=None):
        return _weighted_gradient_fixture(angular, displacement)
    system, orbital, auxiliary = fixture()
    q = np.zeros(3) if gamma else np.array([.13, -.09, .07])
    kpoints = np.array([[.2, -.1, .3], [-.15, .25, -.1]])
    rng = np.random.default_rng(142)
    m_shape = (auxiliary.nbasis, auxiliary.nbasis)
    t_shape = (2, auxiliary.nbasis, orbital.nbasis, orbital.nbasis)
    wm = np.asfortranarray(rng.normal(size=m_shape) + 1j * rng.normal(size=m_shape))
    wt = np.ascontiguousarray(rng.normal(size=t_shape) + 1j * rng.normal(size=t_shape))
    caps = (2**20, 100000, 16 * 2**20, screen)
    gm = core.compute_gdf_sr_metric_gradient_weighted(
        auxiliary, system, q, .6, 5.1, wm, *caps,
    )
    gt = core.compute_gdf_sr_three_center_gradient_weighted(
        orbital, auxiliary, system, q, kpoints, .6, 4.3, 5.1, wt, *caps,
    )
    np.testing.assert_allclose(gm.sum(axis=0), 0., rtol=0, atol=2e-10)
    np.testing.assert_allclose(gt.sum(axis=0), 0., rtol=0, atol=2e-10)
    def energy(displacement):
        sys, orb, aux = fixture(displacement)
        metric = core.compute_gdf_sr_metric(aux, sys, q, .6, 5.1, *caps)
        tensor = core.compute_gdf_sr_three_center(orb, aux, sys, q, kpoints, .6, 4.3, 5.1, *caps)
        return np.array([np.sum(wm * metric).real, np.sum(wt * tensor).real])
    # The f-shell metric weights reach hundreds of Ha/bohr. Use a
    # fourth-order stencil so a larger step resolves the contracted
    # integral differences without an O(h^2) truncation bias.
    h = 1e-4
    for axis in range(3):
        finite_difference = (
            energy((0, axis, -2*h)) - 8*energy((0, axis, -h))
            + 8*energy((0, axis, h)) - energy((0, axis, 2*h))
        ) / (12*h)
        np.testing.assert_allclose([gm[0, axis], gt[0, axis]], finite_difference, rtol=0, atol=2e-8)
    with pytest.raises((ValueError, RuntimeError), match='output byte cap'):
        core.compute_gdf_sr_three_center_gradient_weighted(
            orbital, auxiliary, system, q, kpoints, .6, 4.3, 5.1, wt,
            1, 100000, 16 * 2**20, screen,
        )
    with pytest.raises(TypeError):
        core.compute_gdf_sr_three_center_gradient_weighted(
            orbital, auxiliary, system, q, kpoints, .6, 4.3, 5.1,
            wt.transpose(0, 1, 3, 2), *caps,
        )


@pytest.mark.parametrize('angular,gamma', [(0, True), (2, False), (3, True)])
def test_combined_sr_lr_derivative_matches_its_energy_source(angular, gamma):
    """Differentiate raw M/T, including overlap g0, angular factors and G panels."""
    from itertools import product
    core = vq._vibeqc_core
    system, orbital, auxiliary = _weighted_gradient_fixture(angular, orbital_angular=angular)
    q = np.zeros(3) if gamma else np.array([.13, -.09, .07])
    kpoints = np.array([[.2, -.1, .3], [-.15, .25, -.1]])
    addresses = np.array(list(product(range(-2, 3), repeat=3)) + [
        (3, 0, 0), (-3, 0, 0), (0, 3, 0), (0, -3, 0), (0, 0, 3),
    ])
    vectors = np.ascontiguousarray(addresses @ (2 * np.pi * np.linalg.inv(system.lattice)) + q)
    rng = np.random.default_rng(664)
    m_shape = (auxiliary.nbasis, auxiliary.nbasis)
    t_shape = (2, auxiliary.nbasis, orbital.nbasis, orbital.nbasis)
    wm = np.asfortranarray(rng.normal(size=m_shape) + 1j * rng.normal(size=m_shape))
    wt = np.ascontiguousarray(rng.normal(size=t_shape) + 1j * rng.normal(size=t_shape))
    caps = (2**20, 32 * 2**20, 100000, 0.)
    def derivative(mesh=vectors, weights=(wm, wt)):
        return core.compute_gdf_range_separated_gradient_weighted(
            orbital, auxiliary, system, q, kpoints, mesh, .6, 4.3, 5.1,
            *weights, *caps,
        )
    gradient = derivative()
    np.testing.assert_allclose(gradient.sum(axis=0), 0., rtol=0, atol=2e-10)
    def energy(displacement):
        sys, orb, aux = _weighted_gradient_fixture(angular, displacement, orbital_angular=angular)
        metric, tensor, _ = core.compute_gdf_range_separated_integrals(
            orb, aux, sys, q, kpoints, vectors, .6, 4.3, 5.1, *caps,
        )
        return np.real(np.sum(wm * metric) + np.sum(wt * tensor))
    h = 1e-5
    for axis in range(3):
        expected = (energy((0, axis, h)) - energy((0, axis, -h))) / (2 * h)
        np.testing.assert_allclose(gradient[0, axis], expected, rtol=0, atol=5e-8)
    if gamma:
        # The finite SR zero mode is subtracted even if the caller's primed
        # reciprocal list omits zero; its derivative must not disappear.
        without_zero = vectors[np.linalg.norm(vectors, axis=1) > 1e-12]
        np.testing.assert_allclose(derivative(without_zero), gradient, rtol=0, atol=2e-11)
    else:
        # Relabel atom 1 by a lattice vector and sew the fixed weights by
        # the inverse Bloch phases. The physical weighted derivative is
        # invariant; an origin-centered image sphere cannot satisfy this.
        shifted, orb_shifted, aux_shifted = _weighted_gradient_fixture(
            angular, orbital_angular=angular, image_shift=True,
        )
        translation = np.asarray(system.lattice)[:, 0]
        aux_atoms = np.repeat([0, 1], 2 * angular + 1)
        orb_atoms = np.array([0] * (2 * angular + 1) + [1])
        aux_phase = np.exp(1j * (q @ translation) * aux_atoms)
        metric_phase = aux_phase[:, None] * aux_phase.conj()[None, :]
        tensor_phase = np.empty_like(wt)
        for k, ket in enumerate(kpoints):
            bra_phase = np.exp(1j * ((ket - q) @ translation) * orb_atoms)
            ket_phase = np.exp(-1j * (ket @ translation) * orb_atoms)
            tensor_phase[k] = aux_phase[:, None, None] * bra_phase[None, :, None] * ket_phase[None, None, :]
        shifted_gradient = core.compute_gdf_range_separated_gradient_weighted(
            orb_shifted, aux_shifted, shifted, q, kpoints, vectors, .6, 4.3, 5.1,
            np.asfortranarray(wm / metric_phase), np.ascontiguousarray(wt / tensor_phase), *caps,
        )
        np.testing.assert_allclose(shifted_gradient, gradient, rtol=0, atol=2e-10)
    with pytest.raises((ValueError, RuntimeError), match='workspace byte cap'):
        core.compute_gdf_range_separated_gradient_weighted(
            orbital, auxiliary, system, q, kpoints, vectors, .6, 4.3, 5.1,
            wm, wt, 2**20, 1, 100000,
        )
    with pytest.raises(ValueError, match='finite squared momenta'):
        derivative(np.array([[1e308, 0., 0.]]))


def test_i_auxiliary_short_range_value_and_force_share_generated_limits():
    """Physical i fitting shells need their own value and derivative kernels."""
    core = vq._vibeqc_core
    positions = np.array([[0., 0., 0.], [.7, .2, .3]])
    def fixture(displacement=0.):
        coords = positions.copy()
        coords[0, 0] += displacement
        system = vq.PeriodicSystem(3, np.eye(3)*20., [vq.Atom(1, p) for p in coords])
        molecule = system.unit_cell_molecule()
        orbital = core.BasisSet(molecule, [
            core.ShellInfo(i, 0, True, [.8], [1.], p) for i, p in enumerate(coords)
        ], 'i-aux-orbitals', coefficients_pre_normalized=False)
        aux = core.BasisSet(molecule, [
            core.ShellInfo(i, 6, True, [1.2], [1.], p) for i, p in enumerate(coords)
        ], 'i-aux-fitting', coefficients_pre_normalized=False)
        return system, orbital, aux
    system, orbital, aux = fixture()
    q, k = np.array([.13, -.09, .07]), np.array([[.2, -.1, .3]])
    rng = np.random.default_rng(142)
    wm = np.asfortranarray(rng.normal(size=(26, 26)) + 1j*rng.normal(size=(26, 26)))
    wt = np.ascontiguousarray(rng.normal(size=(1, 26, 2, 2)) + 1j*rng.normal(size=(1, 26, 2, 2)))
    caps = (2**24, 100000, 64*2**20)
    gm = core.compute_gdf_sr_metric_gradient_weighted(aux, system, q, .6, 2., wm, *caps)
    gt = core.compute_gdf_sr_three_center_gradient_weighted(orbital, aux, system, q, k, .6, 2., 2., wt, *caps)
    def values(step):
        sys, orb, fit = fixture(step)
        metric = core.compute_gdf_sr_metric(fit, sys, q, .6, 2., *caps)
        tensor = core.compute_gdf_sr_three_center(orb, fit, sys, q, k, .6, 2., 2., *caps)
        return np.array([np.sum(wm*metric).real, np.sum(wt*tensor).real])
    h = 1e-5
    np.testing.assert_allclose([gm[0, 0], gt[0, 0]], (values(h)-values(-h))/(2*h), atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(gm.sum(axis=0), 0., atol=2e-12)
    np.testing.assert_allclose(gt.sum(axis=0), 0., atol=2e-12)


@pytest.mark.parametrize("derivative_order", [0, 1])
def test_sr_workspace_tracks_linked_engine_team_and_shell_task_limit(derivative_order):
    """A generous source budget can admit 64 workers instead of a fixed ceiling."""
    core = vq._vibeqc_core
    system = vq.PeriodicSystem(3, np.eye(3) * 20, [vq.Atom(1, [0., 0., 0.])])
    mol = system.unit_cell_molecule()
    def basis(count):
        return core.BasisSet(mol, [
            core.ShellInfo(0, 0, True, [1. + .1*i], [1.], [0., 0., 0.])
            for i in range(count)
        ], "workspace-oracle", coefficients_pre_normalized=True)
    orbital, aux = basis(4), basis(8)
    budget = lambda threads: core.gdf_short_range_workspace_bytes(
        orbital, aux, threads, derivative_order, 1,
    )
    one, two, full = budget(1), budget(2), budget(64)
    assert one > 0 and two > one
    assert full == one + 63 * (two - one)
    # Three-center shell tasks: 8 * 4 * 4. More requested threads do not
    # charge idle engines. Metric has only 8 * 8 tasks.
    assert budget(128) == budget(256)
    with pytest.raises(ValueError, match="positive threads"):
        budget(0)


@pytest.mark.parametrize("threads", [1, 4])
def test_sr_linked_workspace_admission_preserves_the_raw_source(threads):
    core = vq._vibeqc_core
    system, orbital, aux = _small_sr_basis()
    old_threads = core.get_num_threads()
    try:
        core.set_num_threads(threads)
        workspace = core.gdf_short_range_workspace_bytes(orbital, aux, threads)
        q = np.array([.07, -.03, .02])
        args = (orbital, aux, system, q, np.zeros((1, 3)), .7, 3., 3.,
                1024**2, 10_000_000)
        admitted = core.compute_gdf_sr_three_center(*args, workspace)
        reference = _sr_three_center_oracle(system, orbital, aux, q, np.zeros(3), .7, 3.)
        np.testing.assert_allclose(admitted[0], reference, atol=2e-11, rtol=0)
        with pytest.raises(ValueError, match="workspace byte cap"):
            core.compute_gdf_sr_three_center(*args, 1)
    finally:
        core.set_num_threads(old_threads)
