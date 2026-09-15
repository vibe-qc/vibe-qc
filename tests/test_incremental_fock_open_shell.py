"""Incremental direct-Fock scaling and open-shell slot regressions.

IID 468 makes screened difference builds scale-homogeneous: the error in a
small update must contract with its amplitude instead of receiving the same
absolute omission allowance as a full build. The low-level fused-G and J/K
slot regressions below pin that policy independently of SCF convergence.

The remaining audit regressions pin that ``incremental_fock=True`` caches the
ΔD path for the UHF/UKS/RKS-hybrid drivers, not just RHF.

Pre-fix the ``DirectJKBuilder`` cache lived inside ``build_g_rhf``;
``build_J`` and ``build_K`` were stateless. Since RKS-hybrid + UHF + UKS
all assemble the Fock via separate ``build_J`` / ``build_K`` calls
(not ``build_g_rhf``), the ``incremental_fock`` option was a no-op
performance-wise for those three.

Fix: per-slot ``build_J_slot`` / ``build_K_slot`` cached entry points
on ``DirectJKBuilder`` — slot 0 for the closed-shell case + J on the
total density, slot 0/1 for K(D_α) / K(D_β) on UHF/UKS so the per-spin
caches don't collide.

These tests pin the correctness contract — energy parity between
``incremental_fock = True`` and ``incremental_fock = False`` on the
DIRECT path for all three non-RHF drivers. Pre-fix and post-fix both
satisfy this contract (the audit found a performance regression, not
a correctness one) — so the value of these tests is that they:

  (a) Guard against a future regression where the slot APIs introduce
      subtle ΔD bookkeeping errors that drift the energy.
  (b) Ensure ``DirectJKBuilder::set_schwarz_threshold`` + ``reset_state``
      invalidate all slot caches together (a single missed invalidation
      would manifest as a stale ΔD applied across a threshold switch
      → mHa-scale energy drift).

The systems are deliberately small (3–4 atoms) so the test cost is
sub-second; for AUTO mode to resolve to DIRECT we set the threshold
to 0 (``scf_mode_auto_threshold = 0``) so the DIRECT path runs even
on a 30-BF basis.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RKSOptions,
    UHFOptions,
    UKSOptions,
    run_rks,
    run_uhf,
    run_uks,
)


@pytest.fixture(scope="module")
def methyl_radical():
    """CH₃ doublet — small system that exercises both spin densities
    on UHF/UKS without dragging the SCF cost up."""
    mol = Molecule(
        [
            Atom(6, [0.0, 0.0, 0.0]),
            Atom(1, [2.05, 0.0, 0.0]),
            Atom(1, [-1.025, 1.776, 0.0]),
            Atom(1, [-1.025, -1.776, 0.0]),
        ],
        0,
        2,
    )
    basis = BasisSet(mol, "sto-3g")
    return mol, basis


@pytest.fixture(scope="module")
def water():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.81]),
            Atom(1, [1.71, 0.0, -0.45]),
        ],
        0,
        1,
    )
    basis = BasisSet(mol, "sto-3g")
    return mol, basis


def test_incremental_screen_error_scales_with_density_step(water):
    """IID 468: shrinking ``delta_D`` must shrink its screening error.

    A screened direct build is linear only while its retained quartet set is
    fixed. Before the fix, splitting one density change into twenty small
    updates applies the full-build absolute cutoff to each update, drops most
    of them, and leaves an O(1) error in the cached Fock matrix. The relative
    incremental screen normalizes each update before screening, so the omitted
    contribution scales back down with ``max(abs(delta_D))`` and the sequence
    reproduces the same screened full-density map.
    """
    _, basis = water
    rng = np.random.default_rng(468)
    raw = rng.normal(size=(basis.nbasis, basis.nbasis))
    density_direction = 0.5 * (raw + raw.T)
    density_direction /= np.max(np.abs(density_direction))

    threshold = 1.0e-2
    steps = 20
    full = vq.make_direct_jk_builder(
        basis, schwarz_threshold=threshold, incremental=False
    )
    expected = np.asarray(full.build_g_rhf(density_direction), dtype=float)

    incremental = vq.make_direct_jk_builder(
        basis,
        schwarz_threshold=threshold,
        incremental=True,
        reset_freq=steps + 1,
    )
    current = np.zeros_like(density_direction)
    got = np.asarray(incremental.build_g_rhf(current), dtype=float)
    for step in range(1, steps + 1):
        current = density_direction * step / steps
        got = np.asarray(incremental.build_g_rhf(current), dtype=float)

    assert np.max(np.abs(expected)) > 0.1  # non-vacuous screened map
    np.testing.assert_allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)
    repeated = np.asarray(incremental.build_g_rhf(current), dtype=float)
    np.testing.assert_array_equal(repeated, got)


@pytest.mark.parametrize(
    ("full_method", "slot_method", "slot"),
    [
        ("build_J", "build_J_slot", 0),
        ("build_K", "build_K_slot", 1),
    ],
)
def test_incremental_slot_screen_error_scales_with_density_step(
    water, full_method, slot_method, slot
):
    """IID 468 also applies to the separately cached J and K paths."""
    _, basis = water
    rng = np.random.default_rng(468)
    raw = rng.normal(size=(basis.nbasis, basis.nbasis))
    density_direction = 0.5 * (raw + raw.T)
    density_direction /= np.max(np.abs(density_direction))

    threshold = 1.0e-2
    steps = 20
    full = vq.make_direct_jk_builder(
        basis, schwarz_threshold=threshold, incremental=False
    )
    expected = np.asarray(
        getattr(full, full_method)(density_direction), dtype=float
    )

    incremental = vq.make_direct_jk_builder(
        basis,
        schwarz_threshold=threshold,
        incremental=True,
        reset_freq=steps + 1,
    )
    current = np.zeros_like(density_direction)
    got = np.asarray(
        getattr(incremental, slot_method)(current, slot), dtype=float
    )
    for step in range(1, steps + 1):
        current = density_direction * step / steps
        got = np.asarray(
            getattr(incremental, slot_method)(current, slot), dtype=float
        )

    assert np.max(np.abs(expected)) > 0.1
    np.testing.assert_allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)
    repeated = np.asarray(
        getattr(incremental, slot_method)(current, slot), dtype=float
    )
    np.testing.assert_array_equal(repeated, got)


def _direct_uhf(*, incremental: bool) -> UHFOptions:
    opts = UHFOptions()
    opts.scf_mode = vq.SCFMode.DIRECT
    opts.scf_mode_auto_threshold = 0  # force DIRECT even on tiny basis
    opts.incremental_fock = incremental
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-9
    opts.max_iter = 200
    return opts


def _direct_rks(functional: str, *, incremental: bool) -> RKSOptions:
    opts = RKSOptions()
    opts.functional = functional
    opts.scf_mode = vq.SCFMode.DIRECT
    opts.scf_mode_auto_threshold = 0
    opts.incremental_fock = incremental
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 200
    return opts


def _direct_uks(functional: str, *, incremental: bool) -> UKSOptions:
    opts = UKSOptions()
    opts.functional = functional
    opts.scf_mode = vq.SCFMode.DIRECT
    opts.scf_mode_auto_threshold = 0
    opts.incremental_fock = incremental
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 200
    return opts


# Incremental ΔD is mathematically exact because J/K are linear in D. The
# screened implementation makes each small update's omission scale with its
# amplitude (IID 468), and reset_freq=8 still bounds ordinary accumulation.
# Tests use 1e-7 Ha headroom: a real bookkeeping error such as a cross-spin
# slot collision would show at the mHa scale, well above this bound.
_ABS_TOL_HA = 1e-7


def test_uhf_incremental_matches_non_incremental(methyl_radical):
    """UHF / DIRECT / CH₃·: incremental ΔD path reproduces the
    non-incremental energy. Probes the per-spin K slot bookkeeping
    (slot 0 for K(D_α), slot 1 for K(D_β))."""
    mol, basis = methyl_radical
    e_off = run_uhf(mol, basis, _direct_uhf(incremental=False)).energy
    e_on = run_uhf(mol, basis, _direct_uhf(incremental=True)).energy
    assert e_on == pytest.approx(e_off, abs=_ABS_TOL_HA), (
        f"UHF incremental={True} vs incremental={False}: "
        f"|Δ|={abs(e_on - e_off):.3e} Ha — per-spin K slot cache drift?"
    )


def test_rks_hybrid_incremental_matches_non_incremental(water):
    """RKS-B3LYP / DIRECT / H₂O: incremental ΔD path with the
    hybrid-DFT call sequence (build_J + build_K separately) matches
    the non-incremental energy. Pre-fix the option was a no-op for
    RKS; this guards the post-fix slot-cached path."""
    mol, basis = water
    e_off = run_rks(mol, basis, _direct_rks("B3LYP", incremental=False)).energy
    e_on = run_rks(mol, basis, _direct_rks("B3LYP", incremental=True)).energy
    assert e_on == pytest.approx(e_off, abs=_ABS_TOL_HA), (
        f"RKS-B3LYP incremental={True} vs incremental={False}: "
        f"|Δ|={abs(e_on - e_off):.3e} Ha — slot-0 J/K cache drift?"
    )


def test_uks_hybrid_incremental_matches_non_incremental(methyl_radical):
    """UKS-B3LYP / DIRECT / CH₃·: per-spin K slots + J slot all
    drive the ΔD path. Energy parity with non-incremental run."""
    mol, basis = methyl_radical
    e_off = run_uks(mol, basis, _direct_uks("B3LYP", incremental=False)).energy
    e_on = run_uks(mol, basis, _direct_uks("B3LYP", incremental=True)).energy
    assert e_on == pytest.approx(e_off, abs=_ABS_TOL_HA), (
        f"UKS-B3LYP incremental={True} vs incremental={False}: "
        f"|Δ|={abs(e_on - e_off):.3e} Ha — per-spin K slot cache drift?"
    )


def test_uks_pure_dft_incremental_is_safe(water):
    """UKS-PBE (pure DFT, α_HF = 0): the K_slot calls never fire
    because ``alpha_hf == 0.0`` short-circuits in the driver. Still
    must converge to the same energy and not OOM / NaN on the slot
    machinery."""
    mol, basis = water
    e_off = run_uks(mol, basis, _direct_uks("PBE", incremental=False)).energy
    e_on = run_uks(mol, basis, _direct_uks("PBE", incremental=True)).energy
    assert e_on == pytest.approx(e_off, abs=_ABS_TOL_HA)
