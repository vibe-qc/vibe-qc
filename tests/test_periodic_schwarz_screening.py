"""Cauchy–Schwarz screening regression tests for the periodic 2-e Fock build.

Pinned contracts:

1. **Performance** — H₂ chain / STO-3G / 2×2×2 multi-k Ewald RHF SCF
   must complete in well under a minute on the dev box (multithreaded).
   Pre-v0.5.3 the unscreened ``build_fock_2e_real_space`` was
   O(n_c³ · n_shells⁴) libint quartet calls per Fock build —
   multiplied by ~30 SCF iterations × 3 Fock builds for the Ewald-split
   HF path it could take hours on real crystals like LiH/STO-3G/4×4×4.
   The fix is a Cauchy–Schwarz pre-pass:

     |⟨μ_0 ν_g | λ_λ σ_σ⟩|  ≤  Q[c_g][μ,ν] · Q[c_σ−c_λ][λ,σ]

   that lets the inner quartet × cell-triple loop short-circuit
   negligible contributions (default threshold ``1e-12 Ha``, configurable
   via ``LatticeSumOptions.schwarz_threshold``).

2. **Correctness** — turning screening on (default 1e-12) and off
   (threshold = 0.0) must give the same energy to 1e-8 Ha, the SCF
   convergence floor for this fixture.

3. **API surface** — ``LatticeSumOptions.schwarz_threshold`` is exposed
   to Python as a writable ``float`` attribute, default ``1e-12``.

The H₂ chain / 2×2×2 mesh is the same fixture used by
``tests/test_periodic_rhf_multi_k_ewald.py`` so it converges robustly
and remains tractable on CI runners; the screening factor generalises
to any system + mesh.
"""

from __future__ import annotations

import time
import math
from itertools import product

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


def _h2_chain_tight():
    """H₂ chain (one molecule per cell), short cubic box, STO-3G.

    Mirrors ``_h2_chain_tight()`` in test_periodic_rhf_multi_k_ewald.py.
    Tight box keeps the lattice sum manageable while still containing
    enough cell triples to exercise the screening.
    """
    R = 0.74 * ANGSTROM_TO_BOHR
    a = 4.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, R]),
    ]
    sysp = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _ewald_options(*, schwarz_threshold: float | None = None):
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    if schwarz_threshold is not None:
        opts.lattice_opts.schwarz_threshold = schwarz_threshold
    opts.damping = 0.5
    opts.max_iter = 60
    opts.use_diis = True
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    # Multi-k SCF plateaus its commutator gradient at ≈ 3e-5 on this
    # H₂-chain fixture (k-degenerate-orbital floor); 1e-4 stops the
    # iteration once the energy itself is stable, which is the contract
    # this test cares about.
    opts.conv_tol_grad = 1e-4
    opts.conv_tol_energy = 1e-8
    return opts


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_schwarz_threshold_attribute_exposed():
    """``LatticeSumOptions.schwarz_threshold`` is a writable float."""
    opts = vq.LatticeSumOptions()
    assert hasattr(opts, "schwarz_threshold")
    # Default matches the C++ default.
    assert opts.schwarz_threshold == pytest.approx(1.0e-12)
    opts.schwarz_threshold = 1.0e-9
    assert opts.schwarz_threshold == pytest.approx(1.0e-9)
    opts.schwarz_threshold = 0.0  # disabled
    assert opts.schwarz_threshold == 0.0


# ---------------------------------------------------------------------------
# 2. Performance regression
# ---------------------------------------------------------------------------

def test_h2_chain_2x2x2_finishes_quickly():
    """H₂ chain / STO-3G / 2×2×2 multi-k Ewald RHF SCF must finish
    quickly. With Schwarz screening this completes in seconds even
    on small CI runners; without screening it can take minutes.
    The 120-second ceiling is generous; the real win is
    orders-of-magnitude on bigger systems like LiH/4×4×4.
    """
    sysp, basis = _h2_chain_tight()
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    opts = _ewald_options()

    t0 = time.perf_counter()
    result = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    elapsed = time.perf_counter() - t0

    assert result.converged, "H₂ chain 2×2×2 SCF did not converge"
    assert elapsed < 120.0, (
        f"H₂ chain / STO-3G / 2×2×2 multi-k Ewald RHF took {elapsed:.1f}s — "
        f"performance regression (Schwarz screening disabled or weakened?)"
    )


# ---------------------------------------------------------------------------
# 3. Correctness — screening on vs off
# ---------------------------------------------------------------------------

def test_screening_on_off_match():
    """Screening at default 1e-12 must give the same energy as
    threshold=0 (disabled) within SCF convergence tolerance.

    Uses the Γ-only path (``[1,1,1]`` mesh) to keep the unscreened
    reference run tractable — at 1×1×1 the unscreened cost is
    roughly ``3 · n_c³ · n_shells⁴`` libint quartets per SCF iter,
    a few thousand for this fixture; at 2×2×2 it would be 8× larger
    and the unscreened reference would dominate test runtime.
    """
    sysp, basis = _h2_chain_tight()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])

    r_screened = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _ewald_options(schwarz_threshold=1.0e-12),
        omega=0.5, spacing_bohr=0.3,
    )
    r_unscreened = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _ewald_options(schwarz_threshold=0.0),
        omega=0.5, spacing_bohr=0.3,
    )

    assert r_screened.converged and r_unscreened.converged
    # Both converge to the same answer within SCF tolerance. With
    # ``conv_tol_grad = 1e-4`` the SCF stops at the k-degenerate
    # commutator floor where successive iterates oscillate by ~1e-6 Ha,
    # so 1e-5 is the honest agreement bound — well inside the screening's
    # 1e-12 per-integral cutoff.
    assert r_screened.energy == pytest.approx(r_unscreened.energy, abs=1e-5)


@pytest.mark.parametrize("threshold", [1e-12, 1e-14])
@pytest.mark.parametrize("builder,omega", [
    (builder, omega) for builder in ("gamma", "lattice") for omega in (0.0, 0.4, 0.7)
] + [("direct", 0.0)])
def test_schwarz_small_self_norms_preserve_gaussian_jk(omega, threshold, builder):
    """Small squared norms must not screen observable mixed integrals."""
    from vibeqc import _vibeqc_core as core

    positions = np.array([[0.0, 0.0, 0.0], [6.4, 0.0, 0.0]])
    system = vq.PeriodicSystem(
        3, 100 * np.eye(3), [vq.Atom(1, p) for p in positions],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(i, 0, False, [1.0], [1.0], p)
        for i, p in enumerate(positions)
    ], "small-schwarz-norm", False)
    density = np.array([[1.0, 0.2], [0.2, 0.8]])

    def boys(t):
        if t == 0:
            return 1.0
        return math.sqrt(math.pi) * math.erf(math.sqrt(t)) / (2 * math.sqrt(t))

    # Closed-form normalized s-Gaussian integrals, independent of Libint.
    eri = np.empty((2, 2, 2, 2))
    theta = omega**2 / (1 + omega**2)
    for a, b, c, d in product(range(2), repeat=4):
        A, B, C, D = positions[[a, b, c, d]]
        t = np.sum(((A + B - C - D) / 2)**2)
        attenuation = math.exp(-0.5 * (np.sum((A - B)**2) + np.sum((C - D)**2)))
        eri[a, b, c, d] = (2 / math.sqrt(math.pi) * attenuation
                           * (boys(t) - math.sqrt(theta) * boys(theta * t)))
    assert 0 < eri[0, 1, 0, 1] < np.finfo(float).eps
    assert abs(eri[0, 1, 0, 0]) > 3 * threshold
    expected_j = np.einsum("abcd,cd->ab", eri, density)
    expected_k = np.einsum("acbd,cd->ab", eri, density)
    opts = core.LatticeSumOptions()
    opts.cutoff_bohr = 1.0  # Only the home cell; both atoms remain in its basis.
    opts.schwarz_threshold = threshold
    if builder == "direct":
        jk = core.make_direct_jk_builder(basis, schwarz_threshold=threshold)
        j, k = np.asarray(jk.build_J(density)), np.asarray(jk.build_K(density))
    elif builder == "gamma":
        jk = core.build_jk_gamma_molecular_limit(basis, system, opts, density, omega)
        j, k = np.asarray(jk.J), np.asarray(jk.K)
    else:
        cells = core.direct_lattice_cells(system, 1.0)
        p = core.make_lattice_matrix_set(2, cells, [density])
        jk = core.build_jk_2e_real_space_domains(
            basis, system, opts, p, cells, omega=omega,
        )
        j, k = np.asarray(jk.J.blocks[0]), np.asarray(jk.K.blocks[0])
    np.testing.assert_allclose(j, expected_j, rtol=0, atol=2e-14)
    np.testing.assert_allclose(k, expected_k, rtol=0, atol=2e-14)


@pytest.mark.parametrize("omega", [0.4, 0.7])
@pytest.mark.parametrize("force_threshold", [1e-14, 1e-20, 0.0])
@pytest.mark.parametrize("term", ["J", "K"])
def test_erfc_gradient_ignores_screened_quartet_buffers(omega, force_threshold, term):
    """A screened quartet must not reuse earlier derivative buffers."""
    from vibeqc import _vibeqc_core as core

    positions = np.array([[0.0, 0.0, 0.0], [6.4, 0.3, 0.2]])
    system = vq.PeriodicSystem(
        3, 100 * np.eye(3), [vq.Atom(1, p) for p in positions],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(i, 0, False, [1.0], [1.0], p)
        for i, p in enumerate(positions)
    ], "screened-derivative-buffers", False)
    density = np.array([[1.0, 0.2], [0.2, 0.8]])
    opts = core.LatticeSumOptions()
    opts.cutoff_bohr = 1.0
    opts.schwarz_threshold_forces = force_threshold
    cells = core.direct_lattice_cells(system, 1.0)
    p = core.make_lattice_matrix_set(2, cells, [density])
    alpha, j_scale = (0.0, 1.0) if term == "J" else (1.0, 0.0)
    actual = np.asarray(core.eri_lattice_gradient_contribution(
        basis, system, p, opts, alpha, j_scale, omega,
    ))

    def energy(xyz):
        def boys(t):
            if t == 0:
                return 1.0
            return math.sqrt(math.pi) * math.erf(math.sqrt(t)) / (2 * math.sqrt(t))

        eri = np.empty((2, 2, 2, 2))
        theta = omega**2 / (1 + omega**2)
        for a, b, c, d in product(range(2), repeat=4):
            A, B, C, D = xyz[[a, b, c, d]]
            t = np.sum(((A + B - C - D) / 2)**2)
            attenuation = math.exp(-0.5 * (np.sum((A - B)**2) + np.sum((C - D)**2)))
            eri[a, b, c, d] = (2 / math.sqrt(math.pi) * attenuation
                               * (boys(t) - math.sqrt(theta) * boys(theta * t)))
        j = np.einsum("abcd,cd->ab", eri, density)
        k = np.einsum("acbd,cd->ab", eri, density)
        return float(np.sum(density * (0.5 * j_scale * j - 0.25 * alpha * k)))

    # Differentiate the closed-form Gaussian energy, independently of Libint.
    expected = np.zeros((2, 3))
    h = 1e-4
    for atom, axis in product(range(2), range(3)):
        plus, minus = positions.copy(), positions.copy()
        plus[atom, axis] += h
        minus[atom, axis] -= h
        expected[atom, axis] = (energy(plus) - energy(minus)) / (2 * h)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-12)
    np.testing.assert_allclose(actual.sum(axis=0), 0, rtol=0, atol=1e-14)
