"""Periodic HUECKEL initial guess -- same-basin validation.

The periodic HUECKEL guess is a Fock-mode GWH lattice guess:

    H_GWH(g)_{uv} = 0.5 K S(g)_{uv} (eps_u + eps_v),   K = 1.75,

with the home-cell diagonal set to the per-AO atomic energies. It is wired
into the closed-shell periodic Gamma RHF and multi-k RHF/RKS drivers. The
closed-shell Python periodic Ewald/GDF/BIPOLE drivers pass the periodic context
into the same lattice Fock helper. The guess must not change the converged SCF
minimum, so HUECKEL and SAD should reach the same energy on small insulating
cells.
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq


def _he_box():
    system = vq.PeriodicSystem(
        3,
        10.0 * np.eye(3),
        [vq.Atom(2, [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_periodic_huckel_gamma_reaches_same_basin_as_sad():
    def energy(guess, Z, a):
        sysp = vq.PeriodicSystem(
            3, a * np.eye(3), [vq.Atom(Z, [0.0, 0.0, 0.0])],
            charge=0, multiplicity=1)
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        r = vq.run_rhf_periodic_gamma(sysp, basis, opts)
        assert r.converged, f"{guess.name} did not converge"
        return r.energy

    for Z, a in [(2, 10.0), (10, 12.0)]:  # He, Ne
        e_huckel = energy(vq.InitialGuess.HUECKEL, Z, a)
        e_sad = energy(vq.InitialGuess.SAD, Z, a)
        assert abs(e_huckel - e_sad) < 1e-7, (
            f"Z={Z}: HUECKEL {e_huckel} vs SAD {e_sad}")


def test_periodic_huckel_multik_reaches_same_basin_as_sad():
    def mk(driver, guess, opts):
        s = vq.PeriodicSystem(3, 10.0 * np.eye(3), [vq.Atom(2, [0, 0, 0])],
                              charge=0, multiplicity=1)
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        km = vq.monkhorst_pack(s, [2, 2, 2])
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        r = driver(s, b, km, opts)
        assert r.converged, f"{driver.__name__}/{guess.name} did not converge"
        return r.energy

    e_rhf_huckel = mk(vq.run_rhf_periodic, vq.InitialGuess.HUECKEL,
                      vq.PeriodicSCFOptions())
    e_rhf_sad = mk(vq.run_rhf_periodic, vq.InitialGuess.SAD,
                   vq.PeriodicSCFOptions())
    assert abs(e_rhf_huckel - e_rhf_sad) < 1e-7

    def _ks():
        o = vq.PeriodicKSOptions()
        o.functional = "lda"
        return o

    e_rks_huckel = mk(vq.run_rks_periodic, vq.InitialGuess.HUECKEL, _ks())
    e_rks_sad = mk(vq.run_rks_periodic, vq.InitialGuess.SAD, _ks())
    assert abs(e_rks_huckel - e_rks_sad) < 1e-7


def test_periodic_huckel_python_gamma_ewald_reaches_same_basin_as_sad():
    """The Python Gamma-Ewald RHF/RKS drivers pass the periodic context needed
    by the HUECKEL Fock-mode seam."""
    def rhf_energy(guess):
        system, basis = _he_box()
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_rhf_periodic_gamma_ewald3d(
            system, basis, opts, auto_optimize_truncation=False, verbose=0)
        assert result.converged, f"RHF/{guess.name} did not converge"
        return result.energy

    def rks_energy(guess):
        system, basis = _he_box()
        opts = vq.PeriodicKSOptions()
        opts.functional = "lda"
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_rks_periodic_gamma_ewald3d(
            system, basis, opts, auto_optimize_truncation=False, verbose=0)
        assert result.converged, f"RKS/{guess.name} did not converge"
        return result.energy

    assert abs(rhf_energy(vq.InitialGuess.HUECKEL)
               - rhf_energy(vq.InitialGuess.SAD)) < 1e-7
    assert abs(rks_energy(vq.InitialGuess.HUECKEL)
               - rks_energy(vq.InitialGuess.SAD)) < 1e-7


def test_periodic_huckel_python_gdf_rhf_reaches_same_basin_as_sad():
    """The Python GDF RHF driver passes the periodic HUECKEL context."""
    def energy(guess):
        system, basis = _he_box()
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_pbc_gdf_rhf(
            system, basis, opts, progress=False, verbose=0)
        assert result.converged, f"GDF RHF/{guess.name} did not converge"
        return result.energy

    assert abs(energy(vq.InitialGuess.HUECKEL)
               - energy(vq.InitialGuess.SAD)) < 1e-7


def test_periodic_huckel_python_bipole_reaches_same_basin_as_sad():
    """The Python BIPOLE RHF/RKS drivers pass the periodic HUECKEL context."""
    def rhf_energy(guess):
        system, basis = _he_box()
        kmesh = vq.monkhorst_pack(system, [1, 1, 1])
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_pbc_bipole_rhf(
            system, basis, kmesh, opts, progress=False, verbose=0)
        assert result.converged, f"BIPOLE RHF/{guess.name} did not converge"
        return result.energy

    def rks_energy(guess):
        system, basis = _he_box()
        kmesh = vq.monkhorst_pack(system, [1, 1, 1])
        opts = vq.PeriodicKSOptions()
        opts.functional = "lda"
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_pbc_bipole_rks(
            system, basis, kmesh, opts, progress=False, verbose=0)
        assert result.converged, f"BIPOLE RKS/{guess.name} did not converge"
        return result.energy

    assert abs(rhf_energy(vq.InitialGuess.HUECKEL)
               - rhf_energy(vq.InitialGuess.SAD)) < 1e-7
    assert abs(rks_energy(vq.InitialGuess.HUECKEL)
               - rks_energy(vq.InitialGuess.SAD)) < 1e-7
