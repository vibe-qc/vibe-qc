"""Periodic Pipek-Mezey localisation at Γ — Wannier increment 2a.

Validates :func:`vibeqc.periodic_localise.localise_periodic_gamma`. The
occupied Bloch orbitals of a Γ-point periodic SCF, Pipek-Mezey-localised, must:

1. be a **unitary** mix of the canonical occupied orbitals (``Uᵀ U = I``);
2. **preserve the occupied density** (``C_loc C_locᵀ = C_occ C_occᵀ``);
3. **increase the PM objective** (genuinely more localised);
4. reproduce the **molecular** localisation in the isolated-molecule-in-a-box
   limit — the molecular-first oracle (CLAUDE.md §14 / §10: the molecular
   method is the trusted reference for its periodic extension).

System: an H₄ linear chain (4 e⁻ → 2 occupied MOs) centred in a 30-bohr box.
H-only and dilute, so the EWALD_3D Γ grid resolves it cleanly — H₂O/STO-3G is
the known Γ-Ewald tight-core xfail (see ``test_periodic_rhf_ewald.py``), so
cores are avoided here. The chain's canonical occupied MOs are delocalised
over all four atoms; PM compresses them into two bond-region orbitals.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import BasisSet, RHFOptions, compute_dipole, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule, compute_overlap
from vibeqc.localise import (
    foster_boys_localise,
    pipek_mezey_localise,
    pipek_mezey_objective,
)
from vibeqc.periodic_localise import _atom_maps, localise_periodic_gamma

BOX = 30.0
CTR = BOX / 2.0  # box centre (bohr)
SPACING = 1.8
# H₄ linear chain along z, uniform spacing, centred in the box:
# z = 12.3, 14.1, 15.9, 17.7 bohr.
_Z = [CTR + SPACING * (k - 1.5) for k in range(4)]
H4_POS = [[CTR, CTR, z] for z in _Z]


def _periodic_opts():
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 80
    opts.use_diis = True
    return opts


def _pm_centers(C_loc, S, amap, atom_pos):
    """Mulliken-charge-weighted Wannier centres — the same formula the module
    uses, replicated here so the comparison is independent of the module."""
    SC = S @ C_loc
    pop_ao = C_loc * SC
    charges = (amap.T @ pop_ao).T
    return charges @ atom_pos


@pytest.fixture(scope="module")
def periodic_h4():
    atoms = [vq.Atom(1, p) for p in H4_POS]
    sysp = vq.PeriodicSystem(3, np.eye(3) * BOX, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, _periodic_opts(), omega=0.5, auto_optimize_truncation=False
    )
    assert r.converged, f"periodic H4 SCF did not converge ({r.n_iter} iters)"
    return {"system": sysp, "basis": basis, "result": r}


@pytest.fixture(scope="module")
def molecular_h4():
    mol = Molecule([Atom(1, p) for p in H4_POS], charge=0, multiplicity=1)
    basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    return {"mol": mol, "basis": basis, "rhf": rhf, "n_occ": mol.n_electrons() // 2}


@pytest.fixture(scope="module")
def wannier(periodic_h4):
    return localise_periodic_gamma(
        periodic_h4["result"], periodic_h4["basis"], periodic_h4["system"]
    )


def test_localisation_is_unitary(wannier):
    U = wannier.U
    np.testing.assert_allclose(U.T @ U, np.eye(U.shape[0]), atol=1e-6)


def test_density_preserved(periodic_h4, wannier):
    """Localisation only rotates within the occupied space — the AO-basis
    occupied projector C_occ C_occᵀ must be invariant."""
    C_occ = np.asarray(periodic_h4["result"].mo_coeffs)[:, : wannier.n_occ]
    P_can = C_occ @ C_occ.T
    P_loc = wannier.C_loc @ wannier.C_loc.T
    np.testing.assert_allclose(P_loc, P_can, atol=1e-7)


def test_objective_increases(wannier):
    # Canonical (delocalised over 4 atoms) ≈ 0.5; localised (≈2 atoms each) ≈ 1.0.
    assert wannier.objective_final > wannier.objective_initial + 0.2


def test_populations_sum_to_one(wannier):
    np.testing.assert_allclose(
        wannier.charges.sum(axis=1), np.ones(wannier.n_occ), atol=1e-8
    )


def test_centers_on_chain_axis(wannier):
    """Centres sit on the chain axis (box centre in x,y), inside the chain in
    z, and symmetric about the box centre."""
    # All atoms lie on x=y=CTR, so the charge-weighted centres must too.
    np.testing.assert_allclose(wannier.centers[:, 0], CTR, atol=1e-6)
    np.testing.assert_allclose(wannier.centers[:, 1], CTR, atol=1e-6)
    z = np.sort(wannier.centers[:, 2])
    assert _Z[0] - 0.5 < z[0] < CTR
    assert CTR < z[-1] < _Z[-1] + 0.5
    assert abs((z[0] + z[-1]) / 2.0 - CTR) < 0.3


def test_matches_molecular_pm(molecular_h4, wannier):
    """Molecular oracle: in a 30-bohr box the periodic Γ occupied subspace ≈
    the isolated molecule's, so PM must give the same objective and centres."""
    rhf, basis, n_occ = (
        molecular_h4["rhf"],
        molecular_h4["basis"],
        molecular_h4["n_occ"],
    )
    C_occ = np.asarray(rhf.mo_coeffs)[:, :n_occ]
    S = np.asarray(compute_overlap(basis))
    amap, atom_pos = _atom_maps(basis)
    C_loc = pipek_mezey_localise(C_occ, S, amap)
    obj_mol = pipek_mezey_objective(C_loc, S, amap)
    cen_mol = _pm_centers(C_loc, S, amap, atom_pos)

    # Measured agreement on this pinned setup (box=30, ω=0.5, default grid):
    # ~1.5e-8 on the objective, ~2e-6 bohr on the centres — the box occupied
    # subspace is numerically identical to the isolated molecule's. Tolerances
    # keep ~50x margin: tight enough to catch a real regression, loose enough
    # to absorb cross-platform float noise.
    assert abs(obj_mol - wannier.objective_final) < 1e-6
    zc_mol = np.sort(cen_mol[:, 2])
    zc_per = np.sort(wannier.centers[:, 2])
    np.testing.assert_allclose(zc_per, zc_mol, atol=1e-4)


def test_rejects_unknown_method(periodic_h4):
    """Unknown criteria fail closed (PM and Boys are the supported pair)."""
    with pytest.raises(ValueError, match="not supported"):
        localise_periodic_gamma(
            periodic_h4["result"],
            periodic_h4["basis"],
            periodic_h4["system"],
            method="edmiston-ruedenberg",
        )


# ---------------------------------------------------------------------------
# Increment 2b — Foster-Boys + true centroids / spreads
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wannier_boys(periodic_h4):
    return localise_periodic_gamma(
        periodic_h4["result"],
        periodic_h4["basis"],
        periodic_h4["system"],
        method="boys",
    )


def test_boys_unitary_and_density_preserved(periodic_h4, wannier_boys):
    U = wannier_boys.U
    np.testing.assert_allclose(U.T @ U, np.eye(U.shape[0]), atol=1e-6)
    C_occ = np.asarray(periodic_h4["result"].mo_coeffs)[:, : wannier_boys.n_occ]
    np.testing.assert_allclose(
        wannier_boys.C_loc @ wannier_boys.C_loc.T, C_occ @ C_occ.T, atol=1e-7
    )


def test_boys_objective_increases(wannier_boys):
    # Σ_i ⟨i|r⟩² rises ~6 on this chain (large absolute baseline from origin 0).
    assert wannier_boys.objective_final > wannier_boys.objective_initial + 1.0


def test_boys_centroids_match_molecular_boys(molecular_h4, wannier_boys):
    """Independent oracle: periodic Γ Boys centroids (from the lattice-multipole
    dipole) match molecular Boys centroids (from the separate molecular
    compute_dipole) in the 30-bohr-box limit. Two different integral routines."""
    rhf, basis, n_occ = (
        molecular_h4["rhf"],
        molecular_h4["basis"],
        molecular_h4["n_occ"],
    )
    C_occ = np.asarray(rhf.mo_coeffs)[:, :n_occ]
    dip = compute_dipole(basis, [0.0, 0.0, 0.0])
    nbf = basis.nbasis
    dipoles = np.zeros((nbf, nbf, 3))
    dipoles[:, :, 0] = np.asarray(dip.x)
    dipoles[:, :, 1] = np.asarray(dip.y)
    dipoles[:, :, 2] = np.asarray(dip.z)
    C_loc = foster_boys_localise(C_occ, dipoles)
    cen_mol = np.einsum("mi,mnc,ni->ic", C_loc, dipoles, C_loc)
    # Measured |dz| ~2e-6 (two distinct integral routines, box=30); 1e-4 margin.
    np.testing.assert_allclose(
        np.sort(wannier_boys.centroids[:, 2]), np.sort(cen_mol[:, 2]), atol=1e-4
    )


def test_spreads_positive_and_symmetric(wannier_boys):
    """Spreads are variances (> 0); the two equivalent H4 Boys orbitals match
    (measured symmetry residual ~5e-13; both 3.043 bohr²)."""
    assert np.all(wannier_boys.spreads > 0.0)
    s = np.sort(wannier_boys.spreads)
    assert abs(s[0] - s[-1]) < 1e-6


def test_centroids_near_charge_centers(wannier_boys):
    """Non-wrapping regime: true centroids ≈ Mulliken charge-weighted centres
    (they coincide to ~2e-6 for this symmetric chain)."""
    np.testing.assert_allclose(
        np.sort(wannier_boys.centroids[:, 2]),
        np.sort(wannier_boys.centers[:, 2]),
        atol=1e-3,
    )
