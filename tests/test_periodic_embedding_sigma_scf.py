"""Tests for the Ishida two-step SCF-level embedding potential.

The substrate mean-field potential ``V_eff`` (``build_substrate_scf_potential``)
promotes Σ_emb from the bare Hcore to the SCF Fock, capturing the large
Hcore→SCF substrate band shift.  ``run_embedded_surface(..., sigma_scf=True)``
wires it into the full pipeline with an SCF-aware contour window.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, BasisSet, PeriodicSystem
from vibeqc.periodic_embedding.region import TAG_REGION_I, TAG_SUBSTRATE
from vibeqc.periodic_embedding.runner import run_embedded_surface
from vibeqc.periodic_embedding.scf2step import (
    SubstrateMeanField,
    _ensure_lat_opts,
    build_substrate_scf_potential,
)
from vibeqc.periodic_embedding.substrate_gf import _build_hk_sk


def _h_slab(n: int = 4, z: int = 1, a: float = 10.0, sp: float = 2.0) -> PeriodicSystem:
    """An n-atom chain normal to a slab surface (vacuum along c)."""
    vac = 30.0
    lat = np.array([[a, 0, 0], [0, a, 0], [0, 0, vac]])
    atoms = [Atom(z, [a / 2, a / 2, 5.0 + i * sp]) for i in range(n)]
    mult = 1 if (n * z) % 2 == 0 else 2
    return PeriodicSystem(dim=3, lattice=lat, unit_cell=atoms, multiplicity=mult)


def test_substrate_scf_potential():
    """V_eff is real-symmetric and nonzero; the SCF substrate spectrum sits
    well above the Hcore spectrum (the electron-electron repulsion shift)."""
    from scipy.linalg import eigh

    sys = _h_slab(4)
    basis = BasisSet(sys.unit_cell_molecule(), "sto-3g")

    smf = build_substrate_scf_potential(sys, basis)
    assert isinstance(smf, SubstrateMeanField)

    n_ao = smf.v_eff.shape[0]
    assert smf.v_eff.shape == (n_ao, n_ao)
    assert np.isrealobj(smf.v_eff)
    assert np.allclose(smf.v_eff, smf.v_eff.T)  # Hermitian (real-symmetric)
    assert np.abs(smf.v_eff).max() > 0.1  # a genuine mean field, not ~0

    assert np.all(np.diff(smf.mo_energies) >= -1e-9)  # ascending
    assert np.isfinite(smf.e_homo)

    # Hcore→SCF shift: SCF eigenvalues are pushed up by the 2e repulsion.
    H0, S0 = _build_hk_sk(sys, basis, np.zeros(3), _ensure_lat_opts(None))
    ev_hcore = eigh(np.real(H0), np.real(S0), eigvals_only=True)
    assert smf.mo_energies.mean() > ev_hcore.mean() + 1.0


def test_sigma_scf_runs_and_differs_from_hcore():
    """sigma_scf=True runs end-to-end with nonzero occupation, its auto
    contour tracks the SCF-shifted bands, and the density differs from the
    Hcore-Σ_emb path."""
    sys = _h_slab(4)
    basis = BasisSet(sys.unit_cell_molecule(), "sto-3g")
    tags = [TAG_SUBSTRATE] * 3 + [TAG_REGION_I]
    common = dict(surface_k_mesh=(1, 1), contour_n_nodes=24, scf_method="one_shot")

    r_hcore = run_embedded_surface(sys, basis, tags, sigma_scf=False, **common)
    r_scf = run_embedded_surface(sys, basis, tags, sigma_scf=True, **common)

    assert r_hcore.occupation > 1e-3
    assert r_scf.occupation > 1e-3

    # The SCF contour window sits far above the Hcore one (~7 Ha shift),
    # i.e. the auto band-edge estimate followed the shifted spectrum.
    assert r_scf.e_fermi > r_hcore.e_fermi + 1.0
    assert r_scf.e_bottom > r_hcore.e_bottom + 1.0

    # Σ_emb changed, so the region-I density changed.
    assert not np.allclose(r_hcore.density_local, r_scf.density_local)


def test_sigma_scf_odd_electron_raises():
    """The substrate RHF is closed-shell only; odd electron count is a
    clear NotImplementedError, not a silent wrong answer."""
    sys = _h_slab(3)  # 3 electrons
    basis = BasisSet(sys.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError):
        build_substrate_scf_potential(sys, basis)
