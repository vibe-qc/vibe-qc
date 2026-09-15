"""Regression gates for the 2026-05-31 periodic gauge-consistency cluster.

Three findings from the 2026-05-30/31 end-to-end audit, all rooted in a
single failure mode: the Hartree J is built in the Ewald-3D gauge while
the electron-nuclear (V_ne), nuclear-nuclear (e_nuc), and/or
electron-electron (Lpq J/K) pieces are summed in a *different*
convention, so the Madelung / charged-cell self-energy fails to cancel.
Energies come back ``converged=True`` with no warning (CLAUDE.md §7,
"converged garbage").

F1 (FIXED, RHF multi-k) — ``run_rhf_periodic_multi_k_ewald3d`` hard-codes
    the Hartree J to the Ewald-3D builder but read V_ne / e_nuc from
    ``lat_opts.coulomb_method``. With a default ``PeriodicRHFOptions()``
    (``coulomb_method=DIRECT_TRUNCATED``) the bare-gauge V_ne / e_nuc
    were paired with the Ewald-3D J, so LiH FCC primitive at Γ converged
    to ~-264.85 Ha (vs the gauge-consistent -7.52 Ha). The driver now
    forces ``EWALD_3D`` at entry, matching the closed-shell Γ siblings
    (which the periodic-SCF chat's A1 milestone already enforces). The
    RKS multi-k and open-shell UHF/UKS Γ + multi-k drivers share the gap
    but their decomposition tests encode the pre-A1 DIRECT bookkeeping;
    extending the enforcement to them is an A1 follow-up from the
    retired periodic-gauge consistency note in git history, not done here.

F3 (FIXED) — ``run_rhf_periodic_gamma_gdf`` (the *default* backend of
    ``run_periodic_job`` for Γ-only RHF/RKS) over-bound in its
    molecular-limit / dim<3 Lpq-J/K branch: V_ne / e_nuc summed nuclei
    out to the default ``nuclear_cutoff_bohr=25`` while the home-cell-only
    Lpq J/K saw only ``cutoff_bohr=15``. H2 in an 18-bohr box came back
    -1.783 Ha vs the isolated-H2 -1.117 Ha. The driver now clamps the
    nuclear cutoff to the GDF integral cutoff in that branch so all three
    Coulomb pieces share one cell set.

F2 (FIXED) — ``run_rhf_periodic_gamma_ewald3d`` was not ω-invariant on
    tight ionic cells (MgO FCC primitive: ω=0.6 -> -242.0 Ha,
    ω=1.0 -> -218.0 Ha, ~24 Ha apart). The audit hypothesised the
    nuclear Ewald α was not slaved to ω; a fixed-density decomposition
    disproved that. The ω-dependence lived in the Hartree
    ``build_j_ewald_3d`` grid/split path. The builder now evaluates J
    directly from the analytical AO-pair FT in the periodic ``G != 0``
    Coulomb convention, so ω is no longer a numerical degree of freedom
    for the total Hartree matrix. The remaining absolute exxdiv-K parity
    gap is tracked separately from this ω-invariance regression.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import CoulombMethod

BOHR = 0.529177210903


def _fcc_primitive(z1: int, z2: int, a_angstrom: float):
    """Two-atom FCC-primitive (rocksalt sublattice) cell + STO-3G basis."""
    a = a_angstrom / BOHR
    A = (a / 2) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    sys = vq.PeriodicSystem(
        3, A, [vq.Atom(z1, [0, 0, 0]), vq.Atom(z2, [a / 2, a / 2, a / 2])]
    )
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    return sys, basis


def _h2_cubic_box(box_bohr: float = 18.0, bond_bohr: float = 1.4):
    A = np.diag([box_bohr, box_bohr, box_bohr])
    sys = vq.PeriodicSystem(
        3,
        A,
        [vq.Atom(1, [0, 0, -bond_bohr / 2]), vq.Atom(1, [0, 0, bond_bohr / 2])],
    )
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    return sys, basis


# --- F3: gamma GDF molecular-limit must not over-bind (default backend) ---

def test_gamma_gdf_molecular_limit_matches_isolated_at_default_cutoffs():
    """Default-cutoff (nuclear_cutoff_bohr=25 > cutoff_bohr=15) Γ-GDF on
    H2 in an 18-bohr box must equal the isolated-molecule energy and the
    matched-cutoff run — i.e. the nuclear cutoff beyond the GDF integral
    cutoff must NOT change the molecular-limit energy.

    Pre-fix the default cutoffs gave -1.783408 Ha (over-bound by summing
    e-n attraction to image nuclei with no compensating e-e repulsion);
    matched cutoffs gave the correct -1.116738 Ha. This is the
    ``run_periodic_job`` default RHF backend, so the bug shipped in
    default periodic jobs.
    """
    sys, basis = _h2_cubic_box(18.0, 1.4)

    # Isolated H2/STO-3G reference (out-of-cell molecular RHF).
    mol = vq.Molecule([vq.Atom(1, [0, 0, -0.7]), vq.Atom(1, [0, 0, 0.7])])
    e_isolated = vq.run_rhf(mol, vq.BasisSet(mol, "sto-3g")).energy

    opts_default = vq.PeriodicRHFOptions()
    opts_default.max_iter = 80
    opts_default.conv_tol_energy = 1e-9
    assert opts_default.lattice_opts.nuclear_cutoff_bohr > (
        opts_default.lattice_opts.cutoff_bohr
    ), "test premise: default nuclear cutoff must exceed the integral cutoff"
    r_default = vq.run_rhf_periodic_gamma_gdf(sys, basis, opts_default, progress=False)

    opts_matched = vq.PeriodicRHFOptions()
    opts_matched.max_iter = 80
    opts_matched.conv_tol_energy = 1e-9
    opts_matched.lattice_opts.nuclear_cutoff_bohr = (
        opts_matched.lattice_opts.cutoff_bohr
    )
    r_matched = vq.run_rhf_periodic_gamma_gdf(sys, basis, opts_matched, progress=False)

    assert r_default.converged
    assert r_default.backend == "native-gamma-gdf"
    # The clamp makes the default-cutoff run identical to the matched run.
    assert r_default.energy == pytest.approx(r_matched.energy, abs=1e-7)
    # ... and equal to the isolated molecule (molecular limit).
    assert r_default.energy == pytest.approx(e_isolated, abs=2e-3)
    # Hard tripwire on the specific over-binding symptom (-1.783 Ha).
    assert r_default.energy > -1.3, (
        f"Γ-GDF molecular-limit H2/18-bohr-box over-bound to "
        f"{r_default.energy:.6f} Ha at the default nuclear cutoff — the "
        "neutral-cell cutoff mismatch (audit F3) has regressed."
    )


# --- F1: multi-k EWALD_3D default opts must be physical (gauge forced) ---

def test_multik_ewald3d_default_opts_forces_gauge_and_is_physical():
    """``run_rhf_periodic_multi_k_ewald3d`` with a default
    ``PeriodicRHFOptions()`` (coulomb_method=DIRECT_TRUNCATED) must force
    EWALD_3D so V_ne / e_nuc share the Ewald-3D J gauge, instead of
    silently converging to a non-physical energy.

    Pre-fix LiH FCC primitive at Γ converged to ~-264.85 Ha; post-fix it
    matches the explicit-EWALD_3D run (~-7.52 Ha). (The residual ~0.8 Ha
    gap to PySCF's Γ exxdiv='ewald' -8.34 Ha is the documented Γ-only
    Madelung/BZ-sampling gap shared with the Γ Ewald sibling — the GDF
    chat's multi-k milestone — not this finding.)
    """
    sys, basis = _fcc_primitive(3, 1, 4.084)  # LiH
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])

    # Pin modest cutoffs + disable the truncation auto-optimiser: this
    # gate tests the *gauge* (coulomb_method), not cutoff convergence,
    # so keep the lattice sum small and deterministic. coulomb_method is
    # left at its library default (DIRECT_TRUNCATED) — the thing under
    # test — for the "default" run.
    def _mk_opts(coulomb_method=None):
        o = vq.PeriodicRHFOptions()
        o.max_iter = 100
        o.conv_tol_energy = 1e-8
        o.use_diis = True
        o.lattice_opts.cutoff_bohr = 12.0
        o.lattice_opts.nuclear_cutoff_bohr = 16.0
        if coulomb_method is not None:
            o.lattice_opts.coulomb_method = coulomb_method
        return o

    opts_default = _mk_opts()
    assert (
        opts_default.lattice_opts.coulomb_method == CoulombMethod.DIRECT_TRUNCATED
    ), "test premise: a fresh options object defaults to DIRECT_TRUNCATED"
    r_default = vq.run_rhf_periodic_multi_k_ewald3d(
        sys, basis, kmesh, opts_default, auto_optimize_truncation=False, progress=False
    )

    opts_ewald = _mk_opts(CoulombMethod.EWALD_3D)
    r_ewald = vq.run_rhf_periodic_multi_k_ewald3d(
        sys, basis, kmesh, opts_ewald, auto_optimize_truncation=False, progress=False
    )

    assert r_default.converged and r_ewald.converged
    # Physical range — excludes the ~-264.85 Ha converged garbage.
    assert -12.0 < r_default.energy < -4.0, (
        f"multi-k EWALD_3D LiH default-opts energy {r_default.energy:.4f} Ha "
        "is non-physical — the DIRECT_TRUNCATED gauge mismatch (audit F1) "
        "has regressed."
    )
    # Forcing the gauge makes default opts identical to explicit EWALD_3D.
    assert r_default.energy == pytest.approx(r_ewald.energy, abs=1e-6)


# --- F2: omega-invariance on tight ionic cell ---

def test_gamma_ewald3d_total_coulomb_is_omega_invariant_on_mgo():
    """ω is an *internal* Ewald split parameter, so the total Coulomb
    energy ``E_J + E_ne + e_nuc`` — a property of the energy expression,
    evaluated here at a single FIXED density — must be ω-independent
    (the driver docstring promises ~µHa over ω∈[0.3, 2.0]).

    This is a fast fixed-density proxy (no SCF) for the pre-fix
    driver-level symptom: ``run_rhf_periodic_gamma_ewald3d`` on MgO FCC
    primitive at Γ gave ω=0.6 → -242.0 Ha vs ω=1.0 → -218.0 Ha
    (~24 Ha apart). The root cause was the FFT-Poisson / real-space
    split in ``build_j_ewald_3d``; the fixed path evaluates Hartree J
    directly from the analytical AO-pair FT with the ``G = 0`` Coulomb
    mode omitted, so changing ω cannot change the total Coulomb energy.
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

    sys, basis = _fcc_primitive(12, 8, 4.207)  # MgO
    lat = LatticeSumOptions()
    lat.coulomb_method = CoulombMethod.EWALD_3D
    lat.cutoff_bohr = 8.0  # small: the ω-gap is FFT/G=0-driven, not SR
    lat.nuclear_cutoff_bohr = 10.0

    k_gamma = np.zeros(3)
    S = np.real(bloch_sum(compute_overlap_lattice(basis, sys, lat), k_gamma))
    S = 0.5 * (S + S.T)
    T = np.real(bloch_sum(compute_kinetic_lattice(basis, sys, lat), k_gamma))
    V = np.real(bloch_sum(compute_nuclear_lattice_dispatch(basis, sys, lat), k_gamma))
    Hcore = 0.5 * ((T + V) + (T + V).T)

    # Any fixed, physically reasonable density exposes the ω-dependence
    # (the charged-cell term scales with the cell electron count, not the
    # SCF solution): use the Hcore-guess closed-shell density.
    w, U = np.linalg.eigh(S)
    keep = w > 1e-7
    X = U[:, keep] / np.sqrt(w[keep])
    _, Cp = np.linalg.eigh(X.T @ Hcore @ X)
    C = X @ Cp
    n_occ = sys.n_electrons() // 2
    C_occ = C[:, :n_occ]
    D = 2.0 * (C_occ @ C_occ.T)
    D = 0.5 * (D + D.T)

    E_ne = float(np.einsum("ij,ij->", D, V))
    e_nuc = float(nuclear_repulsion_per_cell(sys, lat))

    def _total_coulomb(omega):
        J = build_j_ewald_3d(basis, sys, D, omega, lattice_opts=lat, spacing_bohr=0.5)
        E_J = 0.5 * float(np.einsum("ij,ij->", D, J))
        return E_J + E_ne + e_nuc

    e_lo = _total_coulomb(0.6)
    e_hi = _total_coulomb(1.0)
    assert abs(e_lo - e_hi) < 1e-6, (
        f"Γ EWALD_3D total Coulomb varies with ω at fixed D: "
        f"{e_lo:.4f} (ω=0.6) vs {e_hi:.4f} (ω=1.0) Ha."
    )
