"""Tests for vibeqc.periodic_gapw_range_sep — range-separated hybrid GPW/GAPW.

Validates:
- omega_for_functional returns correct (omega, hf_sr, hf_lr) for known functionals
- SR/LR exchange helpers follow the C++ erfc short-range omega convention
- RSGAPW final breakdown reports screened exchange and the augmented-J energy
- SCF entry points exist and import cleanly
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import PlaneWaveGrid
from vibeqc.periodic_gapw_range_sep import omega_for_functional

pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)

# ---------------------------------------------------------------------------
# omega_for_functional — stateless lookup
# ---------------------------------------------------------------------------


def test_omega_for_functional_hse06():
    """HSE06: omega=0.11, SR-HF=0.25, LR-HF=0.0."""
    omega, hf_sr, hf_lr = omega_for_functional("hse06")
    assert omega == pytest.approx(0.11, rel=1e-6)
    assert hf_sr == pytest.approx(0.25, rel=1e-6)
    assert hf_lr == pytest.approx(0.0, abs=1e-12)


def test_omega_for_functional_cam_b3lyp():
    """CAM-B3LYP: omega=0.33, SR-HF=0.19, LR-HF=0.65 (alpha+beta)."""
    omega, hf_sr, hf_lr = omega_for_functional("cam-b3lyp")
    assert omega == pytest.approx(0.33, rel=1e-6)
    assert hf_sr == pytest.approx(0.19, rel=1e-6)
    assert hf_lr == pytest.approx(0.65, rel=1e-6)


def test_omega_for_functional_wb97x():
    """omegaB97X-V: omega=0.30, HF fractions."""
    omega, hf_sr, hf_lr = omega_for_functional("wb97x")
    assert omega == pytest.approx(0.30, rel=1e-6)
    assert hf_sr >= 0.0
    assert hf_lr >= 0.0


def test_omega_for_functional_wb97mv():
    """omegaB97M-V: omega=0.30."""
    omega, hf_sr, hf_lr = omega_for_functional("wb97mv")
    assert omega == pytest.approx(0.30, rel=1e-6)


def test_omega_for_functional_unknown_raises():
    """An unknown functional name raises ValueError."""
    with pytest.raises(ValueError):
        omega_for_functional("nonexistent_func")


# ---------------------------------------------------------------------------
# K split convention
# ---------------------------------------------------------------------------


def _he_case(L: float = 16.0):
    system = core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * L
    system.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    basis = vq.BasisSet(vq.Molecule(list(system.unit_cell), 0, 1), "sto-3g")
    density = np.array([[2.0]])
    return system, basis, density


def test_k_split_helpers_follow_erfc_short_range_convention():
    """The C++ omega kernel is erfc short range; LR is full minus SR."""
    from vibeqc.periodic_gapw_range_sep import (
        _build_k_long_range,
        _build_k_short_range,
    )

    system, basis, density = _he_case()
    lo = core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0
    omega = 0.3
    k_full = np.asarray(
        core.build_jk_gamma_molecular_limit(
            basis,
            system,
            lo,
            density,
            omega=0.0,
        ).K
    )
    k_sr_direct = np.asarray(
        core.build_jk_gamma_molecular_limit(
            basis,
            system,
            lo,
            density,
            omega=omega,
        ).K
    )

    k_sr = _build_k_short_range(basis, system, density, omega)
    k_lr = _build_k_long_range(basis, system, density, omega)

    assert k_sr == pytest.approx(k_sr_direct, rel=0.0, abs=1e-12)
    assert k_sr + k_lr == pytest.approx(k_full, rel=0.0, abs=1e-12)


def test_rsgapw_final_breakdown_uses_screened_exchange_and_augmented_j():
    """The final result reuses the RS-GAPW energy expression."""
    from vibeqc.periodic_gapw_range_sep import run_periodic_rhf_rsgapw

    system, basis, density = _he_case()
    grid = PlaneWaveGrid(system.lattice, 8, 8, 8)
    omega = 0.3
    hf_sr = 0.25
    hf_lr = 0.0

    result = run_periodic_rhf_rsgapw(
        system,
        basis,
        omega=omega,
        hf_sr_fraction=hf_sr,
        hf_lr_fraction=hf_lr,
        grid=grid,
        initial_density=density,
        max_iter=4,
        use_diis=False,
        quiet=True,
    )

    lo = core.LatticeSumOptions()
    lo.cutoff_bohr = 25.0
    k_sr = np.asarray(
        core.build_jk_gamma_molecular_limit(
            basis,
            system,
            lo,
            result.density,
            omega=omega,
        ).K
    )
    expected_exchange = -0.25 * hf_sr * float(
        np.einsum("ij,ij->", result.density, k_sr)
    )

    assert result.converged
    assert result.breakdown.e_hf_exchange == pytest.approx(
        expected_exchange,
        rel=0.0,
        abs=1e-12,
    )
    assert result.energy == pytest.approx(result.breakdown.e_total, abs=1e-12)
    assert result.energy == pytest.approx(
        result.scf_trace[-1]["energy"],
        rel=0.0,
        abs=1e-10,
    )
    assert result.gapw_correction != pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_rsgapw_imports():
    """Range-separated GAPW module imports cleanly."""
    from vibeqc.periodic_gapw_range_sep import (
        RsGapwJBuilder,
        RsJBuilder,
        run_periodic_rhf_rsgapw,
        run_periodic_rks_rsgapw,
    )

    assert callable(run_periodic_rhf_rsgapw)
    assert callable(run_periodic_rks_rsgapw)
