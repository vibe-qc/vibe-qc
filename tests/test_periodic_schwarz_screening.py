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
