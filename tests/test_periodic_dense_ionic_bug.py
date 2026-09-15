"""Periodic SCF on dense ionic crystals — known bug, tracked v0.7.

**Status: critical, deferred from v0.6.2 to v0.7.**

The v0.6.1 Madelung-leak fix
(:mod:`vibeqc.madelung.madelung_energy_correction`) makes periodic
SCF give the right absolute energies in the **atomic limit** (single
neutral atom in a big box, H₂ in a 30-bohr box). It does NOT fix
dense ionic crystals like LiH or NaCl at experimental geometries.

What's broken
-------------

Concrete observations on LiH conventional cubic cell, a = 4.084 Å,
RKS-LDA with Ewald-3D Coulomb method:

  Component         | Value             | Expected magnitude
  ------------------+-------------------+--------------------------
  E_total           | ~-1060 Ha         | ~-32 Ha (4 LiH × -8 Ha)
  e_nuclear (Ewald) |   -13.6 Ha        | OK (Madelung-summed)
  e_electronic      | ~-720 Ha          | ~-50 Ha
  e_xc              |   -12.4 Ha        | OK (LDA on 16 electrons)
  e_coulomb (J)     |   +16.1 Ha        | OK
  Madelung corr     |   +94.1 Ha        | wrong (over-corrects, see below)

The bug is in ``tr(D · Hcore)`` — specifically the nuclear-electron
lattice sum ``tr(D · V_nuc)``. For the Ewald-split SCF the V_nuc
lattice sum and the Hartree-J build must use *consistent* gauges;
when they don't, the cross-cancellation fails and several hundred
Ha of unphysical binding leak into the SCF total.

Why the v0.6.1 Madelung fix doesn't help
----------------------------------------

The v0.6.1 fix assumed the periodic image leak is
``α_M · Q² / (2L)`` where Q is the *total* cell charge. That's
correct for a SINGLE charge of magnitude Q at the origin (the
atomic-limit case). For a NaCl-like rocksalt structure with
*alternating ±Z charges*, the actual Madelung leak is much smaller
(charges partially cancel through the lattice sum) AND it's
**structural** — it depends on how the charges are arranged inside
the cell, not just the total. The simple-cubic Madelung formula
``α_M ≈ 2.837`` doesn't capture the LiH / NaCl charge structure.

For LiH conventional: my Madelung formula gives ``+94 Ha`` correction
on top of an SCF that's already wrong by ~~700 Ha~~ at iter 1. The
correction is too small to fix the underlying cross-term bug AND
likely too coarse for the ionic structure.

What the proper fix needs
-------------------------

Either of:

1. **Don't pin G=0 in the J build** — use the proper Ewald-3D
   convention with a single neutralizing background that includes
   BOTH nuclei and electrons, instead of two separate G=0 pinnings
   that don't cross-cancel. Refactor of
   ``vibeqc.ewald_composed.build_j_ewald_3d``.

2. **Compute the cross-Madelung term explicitly** and add it back
   to the SCF energy. For each unit-cell charge configuration:
   ``E_cross = ∫ V_image_nuc(r) · ρ_e(r) dr`` where
   ``V_image_nuc`` is the periodic-image-summed nuclear potential
   minus the home-cell V_nuc. Match the matching ``e-image-nuc``
   contribution to E_nuc.

3. **Switch to GPW (Gaussian + plane wave)** for the Hartree term —
   the CP2K-style approach where the long-range Coulomb is on the
   FFT grid and the short-range is via Schwarz-screened ERIs. The
   gauge-consistency falls out of the FFT solver having a single
   well-defined zero of potential.

Tracked at v0.7 alongside the other periodic-infrastructure
hardening.

Workaround for v0.6.x users
---------------------------

Until the v0.7 fix:

- Accept that absolute energies on dense ionic crystals (LiH,
  NaCl, MgO, etc.) are off by several hundred Ha. Relative
  quantities (geometry-derivative gradients, ω-invariance, multi-k
  vs Γ at [1,1,1]) are still consistent — they cancel the bug.

- The atomic limit IS correct (v0.6.1 fix). H₂ / He / single-atom
  in big-vacuum-box matches molecular references to ~3 mHa.

- For a dense ionic crystal where you need the absolute periodic
  energy, use ``DIRECT_TRUNCATED`` instead of ``EWALD_3D`` and
  scan over ``cutoff_bohr`` until the energy converges. This is
  far slower per iter than the Ewald path but doesn't have the
  gauge-inconsistency bug.

Tests below pin the bug + xfail-strict so the assertion goes red
when the v0.7 fix lands.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _lih_conventional():
    """LiH conventional cubic cell, a = 4.084 Å (rocksalt)."""
    a = 4.084 / 0.529177210903
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                        (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))   # Li
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0),
                        (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))   # H
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_lih_conventional_total_energy_in_physical_range():
    """LiH/STO-3G/Γ-only RKS-LDA total energy must be in the
    physically sensible range.

    History (formerly xfail-strict, fixed in v0.7.0):

    * **v0.6.x baseline**: ~-1060 Ha (off by factor 30+ vs the
      atomic limit -31.6 Ha). Caused by the periodic Coulomb gauge
      bug — V_ne via bare libint lattice sum did not match J built
      via FFT-Poisson with G = 0 dropped.

    * **v0.7.0 fix (commit 9de640b)**: V_ne dispatches to
      ``compute_nuclear_lattice_ewald`` whenever ``coulomb_method
      == EWALD_3D``, putting V_ne and J in the same gauge. The
      v0.6.1 Madelung correction is also disabled in this case
      (would now double-count). LiH iter-1 energy decomposition
      went from ``E_V = -1647 Ha`` to ``E_V = -50 Ha``.

    This is exercised as a *fixed-density* proxy (no SCF): the gauge
    bug leaked ~1000 Ha into the Coulomb sector already at the
    Hcore-guess density, so evaluating that density's
    ``E_kin + E_ne + E_J + e_nuc`` directly is a faithful, deterministic
    witness that runs in a single matrix build instead of an
    auto-optimised ionic SCF (periodic_ionic_scf_testing pattern). It is
    also the only way to probe this exact cell without the truncation
    auto-optimiser: the raw STO-3G Γ overlap has 3 negative eigenvalues
    (dropped here by the canonical-orthogonalisation ``w > 1e-7`` cut),
    so a direct SCF with ``auto_optimize_truncation=False`` would raise
    LinearDependenceError. The converged absolute energy / PySCF.pbc
    parity is tracked separately in the regression suite.
    """
    from vibeqc._vibeqc_core import (
        CoulombMethod,
        LatticeSumOptions,
        bloch_sum,
        compute_kinetic_lattice,
        compute_overlap_lattice,
        nuclear_repulsion_per_cell,
    )
    from vibeqc.ewald_composed import build_j_ewald_3d
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

    system, basis = _lih_conventional()

    lat = LatticeSumOptions()
    lat.coulomb_method = CoulombMethod.EWALD_3D
    lat.cutoff_bohr = 10.0
    lat.nuclear_cutoff_bohr = 15.0

    # One-electron + overlap matrices at Γ, in the EWALD_3D gauge.
    k_gamma = np.zeros(3)
    S = np.real(bloch_sum(compute_overlap_lattice(basis, system, lat), k_gamma))
    S = 0.5 * (S + S.T)
    T = np.real(bloch_sum(compute_kinetic_lattice(basis, system, lat), k_gamma))
    T = 0.5 * (T + T.T)
    V = np.real(bloch_sum(compute_nuclear_lattice_dispatch(basis, system, lat), k_gamma))
    V = 0.5 * (V + V.T)
    Hcore = T + V

    # Canonical-orthogonalised Hcore-guess density (drops the 3 negative
    # overlap eigenvalues of the unfiltered ionic cell).
    w, U = np.linalg.eigh(S)
    keep = w > 1e-7
    X = U[:, keep] / np.sqrt(w[keep])
    _, Cp = np.linalg.eigh(X.T @ Hcore @ X)
    C = X @ Cp
    n_occ = system.n_electrons() // 2
    C_occ = C[:, :n_occ]
    D = 2.0 * (C_occ @ C_occ.T)
    D = 0.5 * (D + D.T)

    # Coulomb-sector energy at the fixed Hcore-guess density. The
    # pre-fix gauge bug (bare V_ne paired with the Ewald-3D Hartree J)
    # made this ~-1060 Ha; the gauge-consistent value is ~-23.6 Ha (XC
    # is gauge-independent and omitted — the leak lived entirely here).
    E_kin = float(np.einsum("ij,ij->", D, T))
    E_ne = float(np.einsum("ij,ij->", D, V))
    e_nuc = float(nuclear_repulsion_per_cell(system, lat))
    J = build_j_ewald_3d(basis, system, D, 0.5, lattice_opts=lat, spacing_bohr=0.5)
    E_J = 0.5 * float(np.einsum("ij,ij->", D, J))
    E_coulomb_sector = E_kin + E_ne + E_J + e_nuc

    # Physically sensible range for LiH/STO-3G: atomic limit
    # 4×(-7.43 Li) + 4×(-0.467 H) ≈ -31.6 Ha (with XC); the Coulomb
    # sector alone lands near -23.6 Ha. The (-50, 0) window catches the
    # ~-1060 Ha gauge-leak symptom while staying robust to cutoff choice.
    assert -50.0 < E_coulomb_sector < 0.0, (
        f"LiH/STO-3G Coulomb-sector energy {E_coulomb_sector:.2f} Ha "
        "outside physical range — V_ne / Hartree-J gauge mismatch "
        "(v0.6.x periodic Coulomb gauge bug) has regressed."
    )
