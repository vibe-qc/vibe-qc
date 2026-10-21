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
    # "flat" keeps the XC image domain at ``cutoff``; the default grows it
    # past both fixtures' cutoffs (#465).
    return driver.run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), opts, functional=functional,
        use_compcell=False, check_energy_sanity=False, progress=False,
        xc_density_domain=domain, rcut_strategy="flat",
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


@pytest.mark.parametrize("method", ["rks", "uks"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
@pytest.mark.parametrize("external", [False, True])
def test_bipole_terminal_count_uses_returned_density(monkeypatch, method, mesh, external):
    """Public drivers retain their last accepted XC integral on capped returns."""
    import importlib

    module = importlib.import_module(f"vibeqc.pbc_bipole_{method}")
    system, basis = _h2_cube()
    opts = core.PeriodicKSOptions()
    opts.max_iter = 1
    opts.functional = _EXTERNAL if external else "svwn"
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.lattice_opts.nuclear_cutoff_bohr = 8.0
    opts.lattice_opts.becke_image_radius_bohr = 6.0
    opts.becke_image_radius_bohr = 7.0
    opts.grid.n_radial = 16
    opts.grid.n_theta = 7
    opts.grid.n_phi = 12
    selected = (core.PeriodicXCDensityDomain.PERIODIC_LATTICE if external
                else core.PeriodicXCDensityDomain.AUTO)
    name = "build_xc_periodic" + ("_uks" if method == "uks" else "")
    original = getattr(module, name)
    terminal = {}

    def observe(*args):
        contribution = original(*args)
        count = 2 if method == "uks" else 1
        terminal.update(count=float(contribution.n_electrons_grid),
                        periodic=bool(contribution.periodic_density_bras),
                        energy=float(contribution.e_xc),
                        densities=[np.array(d.blocks) for d in args[4:4 + count]],
                        reach=float(args[4 + count].becke_image_radius_bohr))
        return contribution

    monkeypatch.setattr(module, name, observe)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = getattr(module, f"run_pbc_bipole_{method}")(
            system, basis, core.monkhorst_pack(system, list(mesh), use_symmetry=False),
            opts, progress=False, xc_density_domain=selected)
    from vibeqc.periodic_xc_diagnostics import PeriodicXCGridChargeWarning

    assert result.runtime_backend == "pbc-bipole"
    assert result.n_iter == 1
    assert result.xc_grid_electrons == terminal["count"]
    assert result.xc_periodic_density_bras == terminal["periodic"]
    assert result.e_xc == terminal["energy"]
    assert terminal["reach"] == 6.0
    densities = ([result.density_alpha, result.density_beta] if method == "uks"
                 else [result.density])
    for returned, measured in zip(densities, terminal["densities"]):
        np.testing.assert_array_equal(np.asarray(returned.blocks), measured)
    rel = (terminal["count"] - 2.0) / 2.0
    relevant = [w for w in caught if issubclass(w.category, PeriodicXCGridChargeWarning)]
    assert len(relevant) == int(terminal["periodic"] and
                               (not np.isfinite(rel) or abs(rel) > RTOL))
    assert result.xc_grid_electron_error == (rel if terminal["periodic"] else None)
    domain = result.resolved_domain
    assert dict(domain.lattice_2e.values)["becke_image_radius_bohr"] == 6.0
    assert domain.becke_image_radius_bohr == 7.0
    assert opts.lattice_opts.becke_image_radius_bohr == 6.0
    assert opts.becke_image_radius_bohr == 7.0


@pytest.mark.parametrize("max_iter", [1, 40], ids=["capped", "converged"])
@pytest.mark.parametrize(
    "cutoff,image_radius", [(8.0, 6.0), (16.0, 12.0)],
    ids=["truncated", "refined"],
)
def test_kuks_terminal_xc_grid_charge_hook(monkeypatch, max_iter, cutoff, image_radius):
    """KUKS reports the last actual spin-XC evaluation on both return paths."""
    from vibeqc.periodic_xc_diagnostics import (
        PeriodicXCGridChargeWarning, XC_GRID_ELECTRON_RTOL,
    )

    assert XC_GRID_ELECTRON_RTOL == RTOL
    system, basis = _h2_cube()
    opts = core.PeriodicKSOptions()
    opts.max_iter = max_iter
    opts.becke_image_radius_bohr = image_radius
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = 16.0
    terminal = {}
    original_native = core.build_xc_periodic_uks
    original_fold = driver._build_xc_k_from_density_uks

    def observe_native(*args):
        contribution = original_native(*args)
        terminal.update(count=float(contribution.n_electrons_grid),
                        periodic=bool(contribution.periodic_density_bras),
                        energy=float(contribution.e_xc))
        return contribution

    def observe_fold(**kwargs):
        value = original_fold(**kwargs)
        terminal.update(
            alpha=[np.array(d, copy=True) for d in kwargs["density_alpha_k"]],
            beta=[np.array(d, copy=True) for d in kwargs["density_beta_k"]],
            cutoff=float(kwargs["lat_opts"].cutoff_bohr),
            image_radius=float(kwargs["lat_opts"].becke_image_radius_bohr),
        )
        return value

    monkeypatch.setattr(core, "build_xc_periodic_uks", observe_native)
    monkeypatch.setattr(driver, "_build_xc_k_from_density_uks", observe_fold)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = driver.run_kuks_periodic_gdf(
            system, basis, (2, 1, 1), opts, functional=_EXTERNAL,
            check_energy_sanity=False, progress=False, rcut_strategy="flat",
            xc_density_domain=core.PeriodicXCDensityDomain.PERIODIC_LATTICE,
        )

    assert terminal["periodic"]
    assert result.converged == (max_iter == 40)
    assert result.n_iter >= 1
    assert terminal["cutoff"] == cutoff
    assert terminal["image_radius"] == image_radius
    assert result.xc_grid_electrons == terminal["count"]
    relative = (terminal["count"] - 2.0) / 2.0
    assert np.isfinite(relative)
    assert result.xc_grid_electron_error == relative
    assert result.e_xc == terminal["energy"]
    for name in ("alpha", "beta"):
        returned = getattr(result, f"density_{name}")
        assert len(returned) == len(terminal[name]) == 2
        for actual, measured in zip(returned, terminal[name]):
            np.testing.assert_array_equal(np.asarray(actual), measured)
    relevant = [w for w in caught if issubclass(w.category, PeriodicXCGridChargeWarning)]
    assert len(relevant) == int(abs(relative) > RTOL)
    if relevant:
        message = str(relevant[0].message)
        assert message.startswith("multi-k GDF KUKS:")
        assert f"integrates to {terminal['count']:.6f}" in message
        assert f"relative {relative:+.2e}" in message
        assert f"cutoff_bohr={cutoff:g}" in message
        assert f"becke_image_radius_bohr={image_radius:g}" in message
    if max_iter == 40:
        assert (relative < -RTOL) if cutoff == 8.0 else (abs(relative) < RTOL)
