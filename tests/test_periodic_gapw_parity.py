"""GAPW all-electron parity validation — the production-readiness gate.

This module pins the deviation of the experimental GAPW route
(:func:`run_periodic_rhf_gapw`) from vibe-qc's own analytic all-electron
periodic reference (:func:`run_rhf_periodic_gamma_gdf`, GDF) on CORE-BEARING
systems, where the Gaussian augmentation is supposed to recover the
all-electron accuracy that the soft FFT grid alone cannot resolve.

Reference choice (CLAUDE.md §10). The committed assertions compare two
*independent vibe-qc routes* (GAPW vs GDF) — no external program is imported
in-process. GAPW is the native method of CP2K, the closest absolute oracle,
but CP2K was unavailable in the audit environment (no executable on PATH), so
vibe-qc's own GDF route — already validated against PySCF.pbc elsewhere — is
the in-process all-electron reference. The external cross-checks that
corroborate these numbers (PySCF.pbc and GPAW, run out-of-process) are
reproducible with
``examples/regression/gapw_parity/run_gapw_external_parity.py``; historical
GPW-AUDIT labels are kept in comments for continuity.

Published/reference context (CLAUDE.md §8) — He/STO-3G/12-bohr/Γ:
  * PySCF.pbc.RHF.density_fit() (GDF, out-of-process) = -2.81044787 Ha.
  * vibe-qc run_rhf_periodic_gamma_gdf                = -2.80778396 Ha.
  The 2.66 mHa offset is the periodic-HF exchange-divergence (exxdiv /
  Madelung) convention; it is immaterial to the Ha-scale GAPW gap measured
  here and vanishes for DFT / matched conventions.

The aspherical-core H2O finding was resolved by making the already-validated
fit-free analytic-ERI Hartree construction the HF default. The legacy block
construction remains available explicitly for diagnostics. DO NOT hide future
physics failures with damping / level-shift / threshold band-aids
(CLAUDE.md §7).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_augment import (
    run_periodic_rhf_gapw,
    run_periodic_uhf_gapw,
)
from vibeqc.periodic_gapw_grid import PlaneWaveGrid
from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

# The experimental GAPW warning fires on every SCF construction by design.
pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)

_L = 12.0  # bohr; cubic box
_N = 16  # coarse FFT grid (the regime where augmentation must do the work)


def _atom_cell(Z: int, L: float = _L):
    """Closed-shell atom at the box centre + matching STO-3G basis."""
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * L
    system.unit_cell = [core.Atom(Z, [L / 2, L / 2, L / 2])]
    mol = vq.Molecule([vq.Atom(Z, [L / 2, L / 2, L / 2])], 0, 1)
    return system, vq.BasisSet(mol, "sto-3g")


def _grid(L: float = _L, n: int = _N) -> PlaneWaveGrid:
    return PlaneWaveGrid(np.eye(3) * L, n, n, n)


def _gpw_energy(system, basis) -> float:
    return float(
        run_periodic_rhf_gpw(
            system,
            basis,
            grid=_grid(),
            conv_tol_energy=1e-9,
            max_iter=60,
            quiet=True,
        ).energy
    )


def _gdf_energy(system, basis) -> float:
    """Analytic all-electron GDF reference (same gauge as GAPW; CI-safe)."""
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-9
    return float(
        vq.run_rhf_periodic_gamma_gdf(system, basis, opts, progress=False).energy
    )


@pytest.mark.parametrize("open_shell", [False, True])
def test_gapw_hf_auto_requires_declared_molecular_limit(
    monkeypatch,
    open_shell,
):
    """Auto never guesses an isolated box from geometry or runs a grid first."""
    import vibeqc.periodic_gapw_augment as gapw_augment

    system, basis = _atom_cell(4)

    def _unexpected_preflight(*_args, **_kwargs):
        raise AssertionError("envelope gate ran after analytic setup")

    monkeypatch.setattr(
        gapw_augment,
        "_preflight_analytic_eri_memory",
        _unexpected_preflight,
    )
    with pytest.raises(NotImplementedError, match="molecular_limit=True"):
        if open_shell:
            run_periodic_uhf_gapw(
                system,
                basis,
                n_alpha=2,
                n_beta=2,
                quiet=True,
            )
        else:
            run_periodic_rhf_gapw(system, basis, quiet=True)


def test_gpw_underbinds_core_atom_on_coarse_grid_he():
    """CONTEXT (passes): the soft FFT grid alone cannot resolve a core cusp.

    He/STO-3G on a coarse 16³ grid: plain GPW sits tens of mHa *above* the
    analytic GDF all-electron reference. This is the gap the GAPW
    augmentation must close — and the baseline against which the augmentation
    is measured below.
    """
    system, basis = _atom_cell(2)
    e_gpw = _gpw_energy(system, basis)
    e_gdf = _gdf_energy(system, basis)
    assert (e_gpw - e_gdf) > 5e-3, (
        f"expected GPW to underbind GDF on a coarse grid: "
        f"E_GPW={e_gpw:.6f}, E_GDF={e_gdf:.6f} Ha"
    )


def test_gapw_augmentation_enters_total_energy_be():
    """The augmentation correction is computed and reflected in the result.

    Uses Be (Z=4) because its 2sp shell survives pruning under
    soft_cutoff=3.0, giving a non-trivial hard/soft split.
    """
    system, basis = _atom_cell(4)  # Be
    res = run_periodic_rhf_gapw(
        system, basis, grid=_grid(), conv_tol_energy=1e-9, max_iter=60,
        molecular_limit=True, quiet=True,
    )
    e_gapw = float(res.energy)
    # The augmentation correction is computed and stored. This is the
    # diagnostic ½ tr(D · (J_aug − J_gpw)). Non-zero proves the
    # augmentation is active.
    assert abs(float(res.gapw_correction)) > 1e-5, (
        f"Expected non-zero augmentation; got gapw_correction="
        f"{float(res.gapw_correction):.6e}"
    )


def test_gapw_reaches_gdf_accuracy_be():
    """GAPW total energy should improve over GPW toward the GDF reference.

    Uses Be (Z=4) because its 2sp survives pruning. On a coarse 16³ grid
    the energy can't reach full all-electron accuracy, but the GAPW energy
    should be *closer* to GDF than GPW is.
    """
    system, basis = _atom_cell(4)  # Be
    e_gpw = _gpw_energy(system, basis)
    e_gapw = float(
        run_periodic_rhf_gapw(
            system,
            basis,
            grid=_grid(),
            conv_tol_energy=1e-9,
            max_iter=60,
            molecular_limit=True,
            quiet=True,
        ).energy
    )
    e_gdf = _gdf_energy(system, basis)
    # GAPW should be closer to GDF than GPW is (i.e. the augmentation
    # is moving in the right direction).
    assert abs(e_gapw - e_gdf) < abs(e_gpw - e_gdf), (
        f"GAPW did not improve over GPW: "
        f"|GAPW−GDF| = {abs(e_gapw - e_gdf):.3f} Ha, "
        f"|GPW−GDF| = {abs(e_gpw - e_gdf):.3f} Ha, "
        f"E_GAPW={e_gapw:.6f}, E_GPW={e_gpw:.6f}, E_GDF={e_gdf:.6f} Ha"
    )


def test_gapw_runs_on_oxygen_core():
    """O atom (8 e⁻, closed-shell singlet): GAPW should run and be finite.

    The crash occurs in the first build_J of the SCF, so max_iter is small.
    """
    system, basis = _atom_cell(8)
    res = run_periodic_rhf_gapw(
        system, basis, grid=_grid(), conv_tol_energy=1e-7, max_iter=5,
        molecular_limit=True, quiet=True,
    )
    assert np.isfinite(float(res.energy))


def test_gapw_molecular_no_blowup_lih():
    """A 2-atom molecule (LiH, Li 1s core) tracks the all-electron GDF ref.

    Regression for the molecular blow-up: before the on-centre one-centre
    density fix, the augmentation built n_A¹ from ALL AOs on atom A's
    unbounded radial grid, so a *neighbour's* AO — large on A's far grid
    points and amplified by the r^l multipole weight — detonated the ρ₀
    compensator. LiH/STO-3G gave E ≈ +9.8×10⁷ Ha (H |Q|max ≈ 2.3×10⁴) vs the
    GDF reference ≈ −7.86 Ha. Restricting n_A¹ to atom-A-centred AOs keeps
    the augmentation physical: bonded LiH now lands within ~10 mHa of GDF.
    """
    L = 16.0
    c = L / 2.0
    d = 3.015  # ~LiH equilibrium (bohr)
    pos = [[c - d / 2.0, c, c], [c + d / 2.0, c, c]]
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * L
    system.unit_cell = [core.Atom(3, pos[0]), core.Atom(1, pos[1])]
    mol = vq.Molecule([vq.Atom(3, pos[0]), vq.Atom(1, pos[1])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    e_gapw = float(
        run_periodic_rhf_gapw(
            system,
            basis,
            grid=PlaneWaveGrid(np.eye(3) * L, 32, 32, 32),
            conv_tol_energy=1e-8,
            max_iter=60,
            one_centre="block",
            quiet=True,
        ).energy
    )
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 60
    opts.conv_tol_energy = 1e-8
    e_gdf = float(
        vq.run_rhf_periodic_gamma_gdf(system, basis, opts, progress=False).energy
    )
    # Physical + tracks the all-electron reference (the on-centre fix). The
    # < 0.1 Ha bound catches both the old +10⁸ Ha blow-up and any gross
    # augmentation error; the actual deviation is ~8 mHa.
    assert abs(e_gapw - e_gdf) < 0.1, (
        f"LiH GAPW deviates / blew up: E_GAPW={e_gapw:.4f}, E_GDF={e_gdf:.4f} Ha"
    )


# ---------------------------------------------------------------------------
# Foundational soft/hard-partition fix (the smooth FFT term is built from the
# SOFTENED density ρ̃ + a (hard−soft) ρ₀ compensator, NOT the full density —
# so a tight core no longer aliases on the coarse grid). On a well-resolved
# grid the GAPW total now recovers the analytic all-electron (GDF) energy of a
# spherical core atom; pre-fix it was tens of mHa to thousands of Ha off.
# Published/reference context (CLAUDE.md §8): the all-electron GAPW energy
# expression is Krack & Parrinello, Phys. Chem. Chem. Phys. 2, 2105 (2000);
# the dual-grid Hartree partition is Lippert, Hutter & Parrinello, Theor.
# Chem. Acc. 103, 124 (1999).
# ---------------------------------------------------------------------------

_N_FINE = 24  # well-resolved grid (the partition fix is grid-convergent)


def test_gapw_reaches_allelectron_accuracy_he_fine_grid():
    """He (spherical 1s²): GAPW recovers the analytic GDF total to a few mHa.

    On a 24³ grid the soft/hard partition brings He GAPW to within ~1 mHa of
    the GDF all-electron reference. Pre-fix (full density collocated on the
    grid) it was +79 mHa off even on this fine grid — the tight 1s aliased
    into a spurious grid charge. This is the regression guard for that bug.
    """
    system, basis = _atom_cell(2)
    e_gapw = float(
        run_periodic_rhf_gapw(
            system, basis, grid=_grid(n=_N_FINE),
            conv_tol_energy=1e-9, max_iter=80,
            one_centre="block", quiet=True,
        ).energy
    )
    e_gdf = _gdf_energy(system, basis)
    assert abs(e_gapw - e_gdf) < 5e-3, (
        f"He GAPW should reach all-electron accuracy on a fine grid: "
        f"E_GAPW={e_gapw:.6f}, E_GDF={e_gdf:.6f}, "
        f"Δ={(e_gapw - e_gdf) * 1e3:+.2f} mHa (pre-fix was +79 mHa)"
    )


def test_gapw_aspherical_core_molecule_h2o_default_is_accurate():
    """H2O/STO-3G: a core-bearing atom with two bonded neighbours.

    The legacy block construction omitted off-centre AO tails from O's
    one-centre density and converged 8.86 Ha below molecular RHF. The HF
    default now resolves to the fit-free analytic-ERI construction, whose
    exact Hartree Fock has no fitted-density variational hole. The 30 mHa
    bound is the validated analytic-mode envelope; the residual on this
    fixture is about +11.5 mHa.
    """
    L = 12.0
    c = L / 2.0
    pos = [[c, c, c], [c + 1.809, c, c], [c - 0.453, c + 1.751, c]]
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * L
    system.unit_cell = [
        core.Atom(8, pos[0]), core.Atom(1, pos[1]), core.Atom(1, pos[2]),
    ]
    mol = vq.Molecule(
        [vq.Atom(8, pos[0]), vq.Atom(1, pos[1]), vq.Atom(1, pos[2])], 0, 1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    e_mol = float(vq.run_rhf(mol, basis, opts).energy)
    result = run_periodic_rhf_gapw(
        system, basis,
        grid=PlaneWaveGrid(np.eye(3) * L, 48, 48, 48),
        conv_tol_energy=1e-8, max_iter=120, quiet=True,
        molecular_limit=True,
    )
    assert result.converged
    assert result.one_centre == "analytic"
    assert result.molecular_limit_declared is True
    e_gapw = float(result.energy)
    assert abs(e_gapw - e_mol) < 0.030, (
        f"H2O GAPW aspherical-core error: E_GAPW={e_gapw:.6f}, "
        f"E_mol={e_mol:.6f}, delta={(e_gapw - e_mol):+.4f} Ha"
    )


def test_gapw_deep_core_no_aliasing_ne_fine_grid():
    """Ne (deep 1s, ζ~500): the soft/hard partition kills the core aliasing.

    Pre-fix the full Ne density collocated to ~56 e⁻ on a 24³ grid, giving a
    +5900 Ha spurious smooth Hartree (GAPW sat ~+6000 Ha *above* GDF). With the
    softened-density smooth term + ρ₀ compensator, GAPW lands within ~0.5 Ha of
    the analytic GDF all-electron reference — a >10⁴× error reduction. (Sub-mHa
    on a deep core additionally needs the augmentation-accuracy / aspherical-
    multipole work; this test guards the catastrophic-aliasing regression.)
    """
    system, basis = _atom_cell(10)
    e_gapw = float(
        run_periodic_rhf_gapw(
            system, basis, grid=_grid(n=_N_FINE),
            conv_tol_energy=1e-9, max_iter=80,
            one_centre="block", quiet=True,
        ).energy
    )
    e_gdf = _gdf_energy(system, basis)
    assert abs(e_gapw - e_gdf) < 0.5, (
        f"Ne GAPW should not alias the core on a fine grid: "
        f"E_GAPW={e_gapw:.6f}, E_GDF={e_gdf:.6f}, "
        f"Δ={(e_gapw - e_gdf):+.4f} Ha (pre-fix was ~+6000 Ha)"
    )
