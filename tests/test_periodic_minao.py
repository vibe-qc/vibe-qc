"""Periodic MINAO initial guess — same-basin validation.

MINAO projects the ANO-RCC minimal-basis reference density onto the working
basis using lattice-summed Γ overlaps,

    P = S(Γ)^{-1} S_tm(Γ) ,   D = P · D_ref · Pᵀ ,

where S(Γ) is the Γ-folded target overlap and S_tm(Γ) = Σ_g ⟨χ(0)|χ_ref(g)⟩ is
the lattice-summed cross-basis overlap (``compute_minao_density_periodic`` /
``cross_overlap_lattice_gamma``). It is wired into the closed-shell periodic Γ
RHF (``run_rhf_periodic_gamma``) and multi-k RHF/RKS (``run_rhf_periodic`` /
``run_rks_periodic``) drivers as a density-mode guess, placed on the g=0 cell.
The closed-shell Python periodic Ewald/GDF/BIPOLE drivers pass the periodic
context into the same projector.

The guess does not change the converged SCF minimum, so MINAO and SAD reach the
same energy. He and Ne single-atom cells converge cleanly; tighter ionic cells
are a separate SCF-convergence matter, not a guess one.
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


def test_periodic_minao_gamma_reaches_same_basin_as_sad():
    def energy(guess, Z, a):
        sysp = vq.PeriodicSystem(3, a * np.eye(3), [vq.Atom(Z, [0.0, 0.0, 0.0])],
                                 charge=0, multiplicity=1)
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        r = vq.run_rhf_periodic_gamma(sysp, basis, opts)
        assert r.converged, f"{guess.name} did not converge"
        return r.energy

    for Z, a in [(2, 10.0), (10, 12.0)]:  # He, Ne
        e_minao = energy(vq.InitialGuess.MINAO, Z, a)
        e_sad = energy(vq.InitialGuess.SAD, Z, a)
        assert abs(e_minao - e_sad) < 1e-7, (
            f"Z={Z}: MINAO {e_minao} vs SAD {e_sad}")


def test_periodic_minao_multik_reaches_same_basin_as_sad():
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

    # multi-k RHF (PeriodicSCFOptions)
    e_rhf_minao = mk(vq.run_rhf_periodic, vq.InitialGuess.MINAO,
                     vq.PeriodicSCFOptions())
    e_rhf_sad = mk(vq.run_rhf_periodic, vq.InitialGuess.SAD,
                   vq.PeriodicSCFOptions())
    assert abs(e_rhf_minao - e_rhf_sad) < 1e-7

    # multi-k RKS/LDA (PeriodicKSOptions)
    def _ks():
        o = vq.PeriodicKSOptions()
        o.functional = "lda"
        return o
    e_rks_minao = mk(vq.run_rks_periodic, vq.InitialGuess.MINAO, _ks())
    e_rks_sad = mk(vq.run_rks_periodic, vq.InitialGuess.SAD, _ks())
    assert abs(e_rks_minao - e_rks_sad) < 1e-7


def test_periodic_minao_python_gamma_ewald_reaches_same_basin_as_sad():
    """The Python Gamma-Ewald RHF/RKS drivers pass the periodic context needed
    by the MINAO density-mode seam."""
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

    assert abs(rhf_energy(vq.InitialGuess.MINAO)
               - rhf_energy(vq.InitialGuess.SAD)) < 1e-7
    assert abs(rks_energy(vq.InitialGuess.MINAO)
               - rks_energy(vq.InitialGuess.SAD)) < 1e-7


def test_periodic_minao_python_gdf_rhf_reaches_same_basin_as_sad():
    """The Python GDF RHF driver passes the periodic MINAO context."""
    def energy(guess):
        system, basis = _he_box()
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_pbc_gdf_rhf(
            system, basis, opts, progress=False, verbose=0)
        assert result.converged, f"GDF RHF/{guess.name} did not converge"
        return result.energy

    assert abs(energy(vq.InitialGuess.MINAO)
               - energy(vq.InitialGuess.SAD)) < 1e-7


def test_periodic_minao_python_bipole_reaches_same_basin_as_sad():
    """The Python BIPOLE RHF/RKS drivers pass the periodic MINAO context."""
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

    assert abs(rhf_energy(vq.InitialGuess.MINAO)
               - rhf_energy(vq.InitialGuess.SAD)) < 1e-7
    assert abs(rks_energy(vq.InitialGuess.MINAO)
               - rks_energy(vq.InitialGuess.SAD)) < 1e-7
