"""Periodic PATOM initial guess -- closed-shell periodic support and gates."""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _single_atom_system(Z: int, a_bohr: float):
    system = vq.PeriodicSystem(
        3, a_bohr * np.eye(3), [vq.Atom(Z, [0.0, 0.0, 0.0])],
        charge=0, multiplicity=1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_periodic_patom_gamma_reaches_same_basin_as_sad():
    """PATOM uses the Gamma periodic JKBuilder for its in-field step."""
    def energy(guess, Z, a):
        system, basis = _single_atom_system(Z, a)
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_rhf_periodic_gamma(system, basis, opts)
        assert result.converged, f"{guess.name} did not converge"
        return result.energy

    for Z, a in [(2, 10.0), (10, 12.0)]:  # He, Ne
        e_patom = energy(vq.InitialGuess.PATOM, Z, a)
        e_sad = energy(vq.InitialGuess.SAD, Z, a)
        assert abs(e_patom - e_sad) < 1e-7, (
            f"Z={Z}: PATOM {e_patom} vs SAD {e_sad}")


def test_periodic_patom_multik_reaches_same_basin_as_sad():
    """PATOM is wired into the closed-shell multi-k RHF/RKS C++ drivers."""
    def mk(driver, guess, opts):
        system, basis = _single_atom_system(2, 10.0)
        kmesh = vq.monkhorst_pack(system, [2, 2, 2])
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = driver(system, basis, kmesh, opts)
        assert result.converged, (
            f"{driver.__name__}/{guess.name} did not converge")
        return result.energy

    e_rhf_patom = mk(vq.run_rhf_periodic, vq.InitialGuess.PATOM,
                     vq.PeriodicSCFOptions())
    e_rhf_sad = mk(vq.run_rhf_periodic, vq.InitialGuess.SAD,
                   vq.PeriodicSCFOptions())
    assert abs(e_rhf_patom - e_rhf_sad) < 1e-7

    def _ks():
        opts = vq.PeriodicKSOptions()
        opts.functional = "lda"
        return opts

    e_rks_patom = mk(vq.run_rks_periodic, vq.InitialGuess.PATOM, _ks())
    e_rks_sad = mk(vq.run_rks_periodic, vq.InitialGuess.SAD, _ks())
    assert abs(e_rks_patom - e_rks_sad) < 1e-7


@pytest.mark.parametrize("route", ["gpw", "gapw", "uks", "roks"])
def test_multik_gpw_patom_full_exchange_is_seed_only(monkeypatch, route):
    from vibeqc.periodic_gapw_j import run_periodic_rks_gpw_multi_k
    from vibeqc.periodic_corrected_exchange import CorrectedEwaldExchange
    system=vq.PeriodicSystem(3,np.eye(3)*12,[vq.Atom(1,[6,6,5.3]),vq.Atom(1,[6,6,6.7])])
    basis=vq.BasisSet(system.unit_cell_molecule(),'sto-3g')
    mesh=vq.monkhorst_pack(system,[1,1,3])
    seen=[]
    original=CorrectedEwaldExchange.k_space_terms_all_k
    def count(self,*args,**kwargs):
        seen.append(1)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(CorrectedEwaldExchange,'k_space_terms_all_k',count)
    from vibeqc.periodic_gapw_augment import run_periodic_rks_gapw_multi_k
    from vibeqc.periodic_gapw_open_shell import run_periodic_uks_gpw_multi_k, run_periodic_roks_gpw_multi_k
    driver = {'gpw':run_periodic_rks_gpw_multi_k,'gapw':run_periodic_rks_gapw_multi_k,
              'uks':run_periodic_uks_gpw_multi_k,'roks':run_periodic_roks_gpw_multi_k}[route]
    result=driver(system,basis,mesh,functional='lda',initial_guess='PATOM',
        cutoff_ha=8,max_iter=2,quiet=True)
    assert len(seen)==2
    assert result.guess_selection.effective == vq.InitialGuess.PATOM
    assert np.isfinite(result.energy)
    assert result.n_iter==2
