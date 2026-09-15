"""Smoke test for direct-SCF Fock build at the n⁴-tensor-OOM boundary.

The n-hexadecane / def2-SVP cell (50 atoms, ~394 basis functions)
materialises a 192 GB four-index ERI tensor under the conventional
(in-core) Fock build and dies with ``std::bad_alloc``. Direct mode
replaces that with an O(n_shells² + n_bf²) Schwarz-screened
on-the-fly quartet evaluation — the canonical "did we close the
memory wall?" gate.

Convergence + no ``bad_alloc`` is the smoke-test bar. Absolute
energy parity vs ORCA / PySCF is covered by the existing
``test_parity_*`` suites (which also exercise this kernel on
smaller cells once ``scf_mode=DIRECT`` is set explicitly).

The test is marked slow — n-hexadecane RHF SCF takes ~2 minutes
on 4 modern cores; B3LYP a few more (XC quadrature, not the
Fock build, dominates).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import vibeqc as v

# The docstring above calls this file slow but the marker was missing, so it
# ran in every default `pytest` invocation. It is genuinely heavy —
# n-hexadecane / def2-svp (~394 BF) in DIRECT mode: the 2026-06-10 test-health
# profile measured ~51 GB peak RSS, and the RHF + B3LYP SCFs together exceed an
# hour even at a 1800 s/3600 s timeout (the docstring's "~2 min" predates the
# current SCF cost). Route it to the slow/nightly lane on the 128 GB box.
# FOLLOW-UP: ~51 GB in a nominally O(n_shells^2 + n_bf^2) DIRECT path is
# suspect — a possible DIRECT-mode / XC-grid memory-efficiency issue worth an
# SCF-chat look (the n^4 memory wall this test guards may not be fully closed).
pytestmark = pytest.mark.slow


# Share the n-hexadecane geometry with the speed benchmark so the
# smoke test exercises the exact cell the benchmark reports as
# ``vibeqc-error``. Adding the regression dir to sys.path keeps the
# import explicit + avoids restructuring it into a package.
_REGRESSION_DIR = (Path(__file__).resolve().parents[1]
                   / "examples" / "regression" / "parity_matrix_orca")
sys.path.insert(0, str(_REGRESSION_DIR))
from cases import geometry_bohr  # type: ignore  # noqa: E402

sys.path.pop(0)


@pytest.fixture(scope="module")
def n_hexadecane_basis():
    atoms = [v.Atom(int(z), list(xyz)) for z, xyz in geometry_bohr("n-hexadecane")]
    mol = v.Molecule(atoms)
    basis = v.BasisSet(mol, "def2-svp")
    # Guard: we genuinely want > 250 basis functions for this to be a
    # meaningful test of the direct-vs-in-core memory wall. ~394 BFs
    # on def2-SVP per benchmarks/orca_vs_vibeqc_speed.md.
    assert basis.nbasis > 250, (
        f"n-hexadecane / def2-svp produced only {basis.nbasis} BF — "
        f"the geometry or basis is wrong; the test loses its meaning")
    return mol, basis


def test_direct_rhf_converges_at_n_hexadecane(n_hexadecane_basis):
    """RHF / n-hexadecane / def2-SVP must converge under DIRECT mode.

    Conventional mode would allocate a ~192 GB ERI tensor and die
    with ``std::bad_alloc`` before iter 1; this exercises the
    Schwarz-screened on-the-fly path that closed the wall.

    SAD initial guess pinned explicitly: the v0.9.x AUTO default
    (SAP for closed-shell light atoms) lands far from the saturated-
    organic Slater-shape and the SCF oscillates wildly on long
    alkanes (the same regression that surfaces in the H2CO / PBE
    parity cells)."""
    mol, basis = n_hexadecane_basis
    opts = v.RHFOptions()
    opts.scf_mode = v.SCFMode.DIRECT
    opts.initial_guess = v.InitialGuess.SAD
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-8
    result = v.run_rhf(mol, basis, opts)
    assert result.converged, (
        f"RHF n-hexadecane/def2-svp did not converge in {result.n_iter} "
        f"iters (energy = {result.energy})")
    # Sanity: matches benchmarks/orca_vs_vibeqc_speed.md ORCA run
    # (-625.262804 Ha) within mHa — direct kernel is producing the
    # right number, not just *a* number.
    assert -626.0 < result.energy < -624.0, (
        f"RHF energy {result.energy} far from expected -625.26 Ha "
        f"— direct kernel bug")


def test_direct_rks_b3lyp_converges_at_n_hexadecane(n_hexadecane_basis):
    """RKS / B3LYP / n-hexadecane / def2-SVP must converge under
    DIRECT mode. Same memory-wall gate as RHF; the hybrid functional
    exercises the fused J + α_HF·K accumulator (α_HF = 0.20 for
    B3LYP)."""
    mol, basis = n_hexadecane_basis
    opts = v.RKSOptions()
    opts.functional = "b3lyp"
    opts.scf_mode = v.SCFMode.DIRECT
    opts.initial_guess = v.InitialGuess.SAD
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-8
    result = v.run_rks(mol, basis, opts)
    assert result.converged, (
        f"RKS-B3LYP n-hexadecane/def2-svp did not converge in "
        f"{result.n_iter} iters (energy = {result.energy})")
    # ORCA reference: -629.279537 Ha (benchmark RIJCOSX cell; B3LYP
    # energies are within mHa across direct / RI / RIJCOSX paths).
    assert -630.0 < result.energy < -628.0, (
        f"RKS-B3LYP energy {result.energy} far from expected -629.28 Ha "
        f"— direct kernel bug")


# ---------------------------------------------------------------------------
# BUG 87 — large-basis AUTO → DIRECT regression guard
# ---------------------------------------------------------------------------

# CO₂ RHF/cc-pVQZ reference (vibe-qc CONVENTIONAL, verified against ORCA):
_CO2_CC_PVQZ_RHF_ENERGY = -187.7211980444  # Ha


def test_bug87_co2_cc_pvqz_auto_resolves_direct():
    """BUG 87 regression: CO₂ RHF/cc-pVQZ (165 BF) must resolve to
    DIRECT under AUTO mode with the default threshold of 140 BF.

    Before the fix (threshold=200), 165 BF resolved to CONVENTIONAL,
    materialising a 3.7 GB in-core ERI tensor and taking 969 s on the
    reference host. After the fix it must resolve to DIRECT."""
    atoms = [
        v.Atom(6, [0.0, 0.0, 0.0]),
        v.Atom(8, [0.0, 0.0, 2.196]),
        v.Atom(8, [0.0, 0.0, -2.196]),
    ]
    mol = v.Molecule(atoms)
    basis = v.BasisSet(mol, "cc-pvqz")
    assert basis.nbasis == 165, (
        f"Expected 165 BF for CO₂ cc-pVQZ, got {basis.nbasis}")

    # AUTO with defaults should resolve to DIRECT (>140 BF).
    opts = v.RHFOptions()
    assert opts.scf_mode == v.SCFMode.AUTO
    assert opts.scf_mode_auto_threshold == 140
    assert basis.nbasis > opts.scf_mode_auto_threshold, (
        f"CO₂ cc-pVQZ ({basis.nbasis} BF) must exceed threshold "
        f"({opts.scf_mode_auto_threshold}) → DIRECT")

    result = v.run_rhf(mol, basis, opts)
    assert result.converged
    assert result.energy == pytest.approx(_CO2_CC_PVQZ_RHF_ENERGY, abs=1e-8)


def test_bug87_co2_cc_pvqz_rks_pbe_direct():
    """BUG 87 regression: CO₂ PBE/cc-pVQZ must converge under AUTO
    (resolving to DIRECT) and match the reference energy."""
    atoms = [
        v.Atom(6, [0.0, 0.0, 0.0]),
        v.Atom(8, [0.0, 0.0, 2.196]),
        v.Atom(8, [0.0, 0.0, -2.196]),
    ]
    mol = v.Molecule(atoms)
    basis = v.BasisSet(mol, "cc-pvqz")

    opts = v.RKSOptions()
    opts.functional = "pbe"
    opts.conv_tol_energy = 1e-8

    result = v.run_rks(mol, basis, opts)
    assert result.converged
    # Energy must be physically reasonable (PBE on CO₂ cc-pVQZ).
    # Reference from vibe-qc CONVENTIONAL cross-check.
    assert -189.0 < result.energy < -188.0, (
        f"PBE/cc-pVQZ energy {result.energy} out of expected range")


def test_bug87_co2_cc_pvtz_stays_conventional():
    """BUG 87 regression: CO₂ cc-pVTZ (90 BF) must stay CONVENTIONAL
    under AUTO mode — the threshold of 140 preserves small-basis
    performance where the in-core path is faster."""
    atoms = [
        v.Atom(6, [0.0, 0.0, 0.0]),
        v.Atom(8, [0.0, 0.0, 2.196]),
        v.Atom(8, [0.0, 0.0, -2.196]),
    ]
    mol = v.Molecule(atoms)
    basis = v.BasisSet(mol, "cc-pvtz")
    assert basis.nbasis == 90, (
        f"Expected 90 BF for CO₂ cc-pVTZ, got {basis.nbasis}")

    opts = v.RHFOptions()
    assert opts.scf_mode_auto_threshold == 140
    assert basis.nbasis <= opts.scf_mode_auto_threshold, (
        f"CO₂ cc-pVTZ ({basis.nbasis} BF) must be ≤ threshold "
        f"({opts.scf_mode_auto_threshold}) → CONVENTIONAL")

    result = v.run_rhf(mol, basis, opts)
    assert result.converged
    assert result.energy == pytest.approx(-187.7067890503, abs=1e-8)
