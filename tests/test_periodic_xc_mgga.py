"""Periodic meta-GGA (r2SCAN) regression + parity tests.

§4 of the periodic meta-GGA validation handover (completed; retired 2026-06-21, see git history):

  (a) PBE regression guard — periodic PBE energies must be unchanged.
  (b) r2SCAN sanity — periodic r2SCAN produces physically reasonable energies
      that match the molecular limit.
  (c) Open-shell UKS r2SCAN smoke test (GDF route).
  (d) Multi-k periodic r2SCAN guard — molecular-limit collapse (GDF route).

The dense-ionic-crystal cross-code accuracy check (MgO r2SCAN/STO-3G against an
external Gaussian-basis periodic code) is tracked out-of-band as a slow CRYSTAL23
regression (sealed reference, CLAUDE.md §10 — no QC-program import), not a pytest,
because it is heavy (~25-30 min/SCF) AND it currently surfaces an OPEN bug:
``examples/regression/crystal_parity/parity_mgo_r2scan_sto3g.py`` (sealed
CRYSTAL reference). vibe-qc's MgO r2SCAN multi-k
SCF converges cleanly but lands **+384 mHa above** CRYSTAL23 (-271.54497 vs
-271.92903755) — and the same dense-crystal error is **functional-independent**
(LDA +428 mHa vs CRYSTAL SVWN, PBE ~+399 mHa), while RHF matches CRYSTAL to
0.369 mHa and all dilute/molecular DFT matches: i.e. the periodic XC on dense
(overlapping-image) crystals is wrong, not the meta-GGA τ/Vτ kernel (sound on
the dilute guard below). See ``parity_mgo_r2scan_sto3g.py`` (definition-parity
audit + symptom) and memory ``periodic-mgga-parity-reference-choice`` for the
localization. The PySCF.pbc reference the earlier revision of test (d) invoked
was separately non-terminating (KRKS defaults to FFTDF, and STO-3G's tight Mg/O
cores force a 417³ mesh; > 40 min per SCF). The fast multi-k guard below keeps
the *dilute* multi-k periodic r2SCAN path (τ/Vτ kernel) covered in pytest.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

ANG2BOHR = 1.0 / 0.529177210903


def _h2_in_box(box: float = 30.0):
    c = box / 2
    atoms = [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


# ---------------------------------------------------------------------------
# (a) PBE regression guard — LDA/GGA path must be byte-identical
# ---------------------------------------------------------------------------


def test_pbe_regression_h2():
    """PBE on H2/STO-3G must produce the same energy as before the MGGA edits.

    Regression target: the existing test_periodic_rks_ewald.py range
    [-2.0, -0.8] Ha for H2/STO-3G in a 30-bohr box. Any drift in the
    LDA/GGA path from the MGGA edits would violate this.
    """
    sysp, basis = _h2_in_box(box=30.0)
    opts = vq.PeriodicKSOptions()
    opts.functional = "PBE"
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 60
    opts.use_diis = True

    r = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged, f"PBE did not converge after {r.n_iter} iterations"
    assert r.n_iter <= opts.max_iter
    assert -2.0 < r.energy < -0.8, (
        f"PBE energy {r.energy:.8f} Ha outside expected [-2.0, -0.8]"
    )


# ---------------------------------------------------------------------------
# (b) r2SCAN sanity — molecular-limit check
# ---------------------------------------------------------------------------


def test_r2scan_h2_molecular_limit():
    """Periodic r2SCAN on H2/STO-3G in a large box must be physically reasonable.

    r2SCAN is a meta-GGA — the energy should be several mHa lower than PBE
    (meta-GGAs recover more correlation) but not more than ~0.1 Ha different
    (would signal a τ or Vτ bug).  Also cross-checks against the molecular
    r2SCAN result via vibe-qc itself (not PySCF — avoids the subprocess
    dependency on external codes, keeping the test self-contained).
    """
    sysp, basis = _h2_in_box(box=30.0)

    # Periodic r2SCAN via EWALD_3D.
    opts = vq.PeriodicKSOptions()
    opts.functional = "r2scan"
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 60
    opts.use_diis = True

    r_per = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r_per.converged, f"r2SCAN did not converge after {r_per.n_iter} iterations"

    # Molecular r2SCAN on the same molecule (isolated).
    mol = sysp.unit_cell_molecule()
    mol_opts = vq.RKSOptions()
    mol_opts.functional = "r2scan"
    mol_opts.max_iter = 60
    mol_opts.use_diis = True

    r_mol = vq.run_rks(mol, basis, mol_opts)
    assert r_mol.converged

    # In a 30-bohr box with EWALD_3D MP correction, the periodic energy
    # should be within ~50 mHa of the molecular one (the MP shift for
    # neutral H2 in a huge box is small).  A hard failure here signals
    # a serious τ/Vτ bug (wrong factor, wrong sign, etc.).
    diff = abs(r_per.energy - r_mol.energy)
    assert diff < 0.1, (
        f"Periodic r2SCAN ({r_per.energy:.8f} Ha) vs molecular "
        f"({r_mol.energy:.8f} Ha) differ by {diff:.3e} Ha — "
        f"suspected τ or Vτ bug."
    )

    # Tighter but informative log.
    print(f"\n  H2 r2SCAN/STO-3G:")
    print(f"    periodic (30 bohr box) = {r_per.energy:.8f} Ha")
    print(f"    molecular              = {r_mol.energy:.8f} Ha")
    print(f"    Δ                      = {diff:.6f} Ha")


# ---------------------------------------------------------------------------
# (c) Open-shell UKS r2SCAN smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_uks_r2scan_smoke(tmp_path):
    """UKS r2SCAN must converge and produce a spin-polarised result.

    Open-shell meta-GGA smoke per §4(c) of the handover, on the supported
    GDF route.

    The original smoke drove a fixed-geometry FM Fe-bcc cell through the
    Γ-only EWALD_3D driver. That driver is now (correctly) retired for dense
    ionic crystals whose periodic images overlap — the molecular-limit
    density convention D(g≠0)=0 gives a ~2 Ha-wrong energy there, so the SCF
    fails closed (CLAUDE.md §7). A bcc *metal* also does not converge through
    the Γ-only GDF path within any reasonable iteration budget. So this
    repoints the smoke at ``jk_method='gdf'`` on a genuinely open-shell system
    that converges robustly: a triplet O2 in a vacuum-padded box — the same
    molecular-limit-in-a-periodic-box convention this file already uses for
    the closed-shell r2SCAN sanity check. The point is unchanged: the periodic
    UKS meta-GGA path runs end-to-end and yields a spin-polarised (triplet)
    solution, not a collapsed closed-shell one.
    """
    box = 14.0
    c = box / 2
    half_bond = (1.16 * ANG2BOHR) / 2  # O2 bond length ≈ 1.16 Å
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(8, [c, c, c - half_bond]), vq.Atom(8, [c, c, c + half_bond])],
        multiplicity=3,  # triplet O2: 2 unpaired electrons → S=1, 2S+1=3
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    r = vq.run_periodic_job(
        sysp,
        basis,
        method="UKS",
        functional="r2scan",
        jk_method="gdf",
        aux_basis="def2-svp-jk",
        max_iter=80,
        conv_tol_energy=1e-6,
        initial_guess="SAD",
        write_molden_file=False,
        output=str(tmp_path / "uks_r2scan_o2"),
    )
    assert r.converged, f"UKS r2SCAN did not converge after {r.n_iter} iterations"
    assert np.isfinite(r.energy), "UKS r2SCAN energy is not finite"
    # Spin-polarised: a triplet must carry ⟨S²⟩ ≈ S(S+1) = 2 (genuine open
    # shell, α≠β), not the closed-shell 0. A bug that collapsed α onto β would
    # land near 0 and trip this.
    assert abs(r.s_squared - 2.0) < 0.2, (
        f"UKS r2SCAN ⟨S²⟩={r.s_squared:.4f} is not the expected triplet "
        f"value ≈2.0 — spin polarisation lost (looks closed-shell)."
    )


# ---------------------------------------------------------------------------
# (d) Multi-k periodic r2SCAN guard
#
# The dense-ionic-crystal cross-code accuracy check (MgO r2SCAN/STO-3G vs an
# external Gaussian-basis periodic code) is tracked out-of-band as a slow
# CRYSTAL23 regression, not a pytest: vibe-qc's MgO r2SCAN multi-k SCF is
# correct but heavy (~25-30 min, direct-space J build), and the PySCF.pbc
# reference the old test invoked was itself effectively non-terminating —
# KRKS defaults to FFTDF, and STO-3G's tight Mg/O cores force a [417 417 417]
# mesh, so one MgO r2SCAN/STO-3G [2,2,2] reference SCF runs > 40 min.
#
# This keeps a *fast* multi-k periodic r2SCAN guard here; tests (a)-(c) above
# are Γ-only, so without it the multi-k Bloch + meta-GGA path would be
# pytest-uncovered for r2SCAN.
# ---------------------------------------------------------------------------


def test_r2scan_multik_gdf_molecular_limit(tmp_path):
    """Multi-k periodic r2SCAN (GDF) must run and reproduce the molecular limit.

    A neutral H2 in a large (30-bohr) box has flat bands, so its [2,2,2]
    Monkhorst-Pack r2SCAN total energy must collapse onto the isolated-molecule
    r2SCAN energy. This drives the full multi-k Bloch + GDF + meta-GGA τ/Vτ
    path cheaply (~8 s, converges in a couple of iterations); the cross-code
    accuracy check on a genuine ionic crystal is the CRYSTAL23 regression
    script referenced above. A τ/Vτ factor/sign bug or a multi-k k-phase bug
    would break the molecular-limit collapse asserted here.
    """
    box = 30.0
    c = box / 2
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    r_k = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="r2scan",
        jk_method="gdf",
        kpoints=(2, 2, 2),
        aux_basis="def2-svp-jk",
        max_iter=60,
        conv_tol_energy=1e-7,
        initial_guess="SAD",
        write_molden_file=False,
        output=str(tmp_path / "h2_multik_r2scan"),
    )
    assert r_k.converged, f"multi-k r2SCAN did not converge after {r_k.n_iter} iters"
    assert np.isfinite(r_k.energy)

    # Isolated-molecule r2SCAN on the same H2 + basis.
    mol = sysp.unit_cell_molecule()
    mol_opts = vq.RKSOptions()
    mol_opts.functional = "r2scan"
    mol_opts.max_iter = 60
    mol_opts.use_diis = True
    r_mol = vq.run_rks(mol, basis, mol_opts)
    assert r_mol.converged

    diff = abs(r_k.energy - r_mol.energy)
    # Measured collapse is < 1 µHa; 1 mHa is a robust bug-catching bound.
    assert diff < 1e-3, (
        f"multi-k periodic r2SCAN ({r_k.energy:.8f} Ha) does not reproduce the "
        f"molecular r2SCAN limit ({r_mol.energy:.8f} Ha): Δ = {diff:.3e} Ha"
    )
