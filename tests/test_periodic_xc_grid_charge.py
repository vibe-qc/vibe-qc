"""The multi-k GDF KS drivers report a truncated XC grid density (vibe-qc#385).

Runner 2.5's independent recompute for #213 found that at modest lattice
cutoffs the periodic XC grid density loses electrons: on the H2 8-bohr cube
(STO-3G, (2,1,1) mesh) at cutoff 8 / image radius 6 bohr the grid integral is
1.9925 for N = 2, and E_xc moves by 1.5e-2 Ha for a quadratic functional,
with no warning. The drivers now record the grid integral of the final XC
evaluation on the result and warn when it misses N by more than
``XC_GRID_ELECTRON_RTOL``; a converged image reach stays silent.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc import periodic_k_gdf as driver

# Relative tolerance of the check (vibeqc.periodic_xc_diagnostics). The
# diagnostics module is imported inside the tests that need it, so that on a
# tree without it the truncation test fails for the issue's reason (no
# warning) rather than at import.
RTOL = 1e-3


class _Quadratic:
    """External all-point functional E = 1/2 (int rho)^2 (#213's provider)."""

    def __call__(self, features):
        w = np.asarray(features["grid_weights"], dtype=float)
        rho = (np.asarray(features["rho_alpha"], dtype=float)
               + np.asarray(features["rho_beta"], dtype=float))
        integral = float(w @ rho)
        z3 = np.zeros((w.size, 3))
        z = np.zeros(w.size)
        return {"energy": 0.5 * integral * integral,
                "v_rho_alpha": integral * w, "v_rho_beta": integral * w,
                "v_grad_alpha": z3, "v_grad_beta": z3,
                "v_tau_alpha": z, "v_tau_beta": z}


_EXTERNAL = "row4-385-quadratic"
vq.define_external_functional(_EXTERNAL, _Quadratic(),
                              hf_exchange_fraction=0.0,
                              capability_version=1,
                              required_grid_profile="")


def _h2_cube():
    system = vq.PeriodicSystem(
        3, np.eye(3) * 8.0,
        [vq.Atom(1, [4.0, 4.0, 3.3]), vq.Atom(1, [4.0, 4.0, 4.7])],
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _run(functional, cutoff, image_radius):
    system, basis = _h2_cube()
    opts = core.PeriodicKSOptions()
    opts.max_iter = 40
    opts.becke_image_radius_bohr = image_radius
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = max(cutoff, 16.0)
    # The external provider runs on the explicit finite-torus domain, as in
    # #213's fixture; built-in functionals keep the driver's default domain.
    domain = (core.PeriodicXCDensityDomain.PERIODIC_LATTICE
              if functional == _EXTERNAL
              else core.PeriodicXCDensityDomain.AUTO)
    return driver.run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), opts, functional=functional,
        use_compcell=False, check_energy_sanity=False, progress=False,
        xc_density_domain=domain,
    )


@pytest.mark.parametrize("functional", [_EXTERNAL])
def test_truncated_xc_image_reach_warns(functional):
    with pytest.warns(RuntimeWarning, match="integrates to"):
        result = _run(functional, cutoff=8.0, image_radius=6.0)
    assert result.xc_grid_electrons < 2.0 * (1.0 - RTOL)
    assert result.xc_grid_electron_error < -RTOL


@pytest.mark.parametrize("functional", [_EXTERNAL, "lda"])
def test_converged_xc_image_reach_is_silent(functional):
    from vibeqc.periodic_xc_diagnostics import (
        XC_GRID_ELECTRON_RTOL,
        PeriodicXCGridChargeWarning,
    )

    assert XC_GRID_ELECTRON_RTOL == RTOL
    with warnings.catch_warnings():
        warnings.simplefilter("error", PeriodicXCGridChargeWarning)
        result = _run(functional, cutoff=16.0, image_radius=12.0)
    assert result.converged
    assert abs(result.xc_grid_electrons - 2.0) / 2.0 < RTOL
    assert abs(result.xc_grid_electron_error) < RTOL


def test_check_ignores_molecular_limit_density():
    from vibeqc.periodic_xc_diagnostics import (
        PeriodicXCGridChargeWarning,
        check_xc_grid_electron_count,
    )

    diag = {"n_electrons_grid": 1.5, "periodic_density_bras": False}
    with warnings.catch_warnings():
        warnings.simplefilter("error", PeriodicXCGridChargeWarning)
        assert check_xc_grid_electron_count(diag, 2.0, context="t") is None
    diag["periodic_density_bras"] = True
    with pytest.warns(PeriodicXCGridChargeWarning):
        rel = check_xc_grid_electron_count(diag, 2.0, context="t")
    assert rel == pytest.approx(-0.25)
