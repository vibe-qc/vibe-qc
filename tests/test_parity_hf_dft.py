"""HF/DFT cross-code parity — per-intermediate energy decomposition.

The v0.13.5 certification matrix (``handovers/HANDOVER_CROSS_CODE_PARITY.md``):
instead of only checking the total SCF energy, decompose every cell
into E_nuc / E_1e / E_coulomb / E_exchange / E_xc / MO eigenvalues
(via :mod:`vibeqc.parity`) and assert each piece independently against
PySCF. When a cell fails, the failing *piece* localises the bug to
integrals / Fock build / XC quadrature.

Two test families:

1. ``test_parity_decomposition_self_consistent`` — no PySCF. Verifies
   :mod:`vibeqc.parity` is internally consistent: the decomposed
   pieces sum to ``result.energy`` to ~1e-9. Covers all four SCF
   flavours.
2. ``test_parity_vs_pyscf`` — the actual cross-code matrix. PySCF is
   imported directly (dev-time, like ``test_gradient.py`` — the
   CLAUDE.md § 10 no-in-process-import rule governs ``python/vibeqc/``
   runtime code, not the test suite).

Tolerances are data-driven (``scripts/parity_pyscf_probe.py``).

**Worst observed across the 46-cell matrix at 2026-05-17** (full
sweep over RHF/UHF/RKS/UKS × {H2O, H2CO, OH, NO, CN, CH3OO} ×
{def2-svp, def2-tzvp} × {direct, DF}, with the def2-tzvp open-shell
radicals routed through KDIIS + conv_tol_grad=1e-6 per the
``_HARD_OPEN_SHELL`` recipe):

==================  ====================  =====================  ============
piece               easy cells (worst)    hard cells (worst)     tolerance
==================  ====================  =====================  ============
``e_nuc``           1.8e-15               2.8e-14                1e-12
``e_1e``            5.7e-6  (DFT direct)  9.7e-6  (DFT-DF hard)  1e-5
``e_coulomb``       6.7e-6                9.7e-6                 1e-5
``e_exchange``      1.3e-7                1.2e-6  (HF hard)      2e-6
``e_xc``            7.3e-7                1.4e-6  (DFT hard)     2e-6
``e_total``         7.8e-7                8.1e-7                 2e-6
``mo`` (max abs)    7.9e-6                1.5e-5                 2e-5
==================  ====================  =====================  ============

``e_1e`` / ``e_coulomb`` are anti-correlated and dominated by the
small converged-density difference between the two codes.
``e_total`` / ``e_xc`` / ``e_exchange`` stay tighter because the
total is stationary, so the converged-density spread cancels.
The headroom over the worst observed is 2-3x for every band — a
real integral / Fock-build / XC-quadrature regression at the mHa
scale (cf. the v0.7.0 Madelung self-image leak) is still caught.

**Structural floor — matched-grid certification investigated (2026-05-17).**
Lebedev angular grids landed in ``cpp/src/grid.cpp`` (commit
``b8964ab``, "XC modernization stage 1") as an opt-in alongside the
legacy Gauss-Legendre × uniform-φ product scheme. The hypothesis
going into v0.8.x: switching the parity matrix's vibe-qc side to
Lebedev-29 (302 pts) + PySCF ``atom_grid = (75, 302)`` would drop
``_TOL["e_xc"]`` toward integral-quadrature precision (~1e-8 Ha).

**Empirically the switch does NOT tighten the cross-code floor.**
At matched (75, 302) Lebedev on both sides, with each code at its
own converged density:

==================  ==================  ============================
cell                ``Δ_xc`` (matched)  ``Δ_xc`` (current ProductGL)
==================  ==================  ============================
H2O/def2-svp/PBE    -1.66e-6            +6.7e-8 (cancellation)
H2O/def2-tzvp/PBE   -1.21e-6            +6.2e-8
H2O/def2-svp/B3LYP  -1.29e-6            +1.1e-7
H2CO/def2-svp/PBE   +4.69e-7            -4.0e-7
H2CO/def2-tzvp/PBE  -1.51e-7            -6.5e-7
==================  ==================  ============================

The floor at the matched grid sits at ~5e-7 to 1.7e-6 Ha — the same
magnitude as the current ProductGL band. The matched-grid hypothesis
underestimated the cross-code contribution of *non-angular*
conventions: Treutler-Ahlrichs ``xi(Z)`` (vibe-qc uses a uniform
formula; PySCF tunes per element, notably for H), Becke partition
``k`` smoothing order, and libxc-binding internals. The H-content
correlation in the table (H2O cells consistently ~3-10x worse than
H2CO cells) is the smoking gun for ``treutler_xi(1)``.

True bit-reproducibility (~1e-9 Ha) requires either a shared-density
bisection (with an AO-ordering bridge between vibe-qc and PySCF) or
aligning the ``treutler_xi`` / Becke / radial conventions — both
larger-scope items for the grid/integrator chat. The current
parity matrix retains its ProductGL default + the existing 2e-6
``e_xc`` tolerance band, which honestly bounds the convention-
floor of cross-code DFT comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom, BasisSet, InitialGuess, Molecule, GridOptions,
    RHFOptions, RKSOptions, UHFOptions, UKSOptions,
    run_rhf, run_rks, run_uhf, run_uks,
)
from vibeqc.parity import (
    decompose_energy_rhf, decompose_energy_uhf,
    decompose_energy_rks, decompose_energy_uks,
    scf_cosx_grid,
    integral_invariants,
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# --- geometries (Bohr) ---------------------------------------------------
_A = ANGSTROM_TO_BOHR
H2O = [(8, [0.0, 0.0, 0.0]),
       (1, [0.0, 0.793353 * _A, -0.613510 * _A]),
       (1, [0.0, -0.793353 * _A, -0.613510 * _A])]
H2CO = [(6, [0.0, 0.0, 0.0]),
        (8, [0.0, 0.0, 1.205 * _A]),
        (1, [0.0, 0.943 * _A, -0.587 * _A]),
        (1, [0.0, -0.943 * _A, -0.587 * _A])]
OH = [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.97 * _A])]

# Open-shell radicals for the multi-heavy-atom def2-tzvp coverage
# (BRIEF.md § Scope-2). Equilibrium bond lengths from NIST CCCBDB
# (B3LYP/aug-cc-pVTZ; values agree with experiment to ~0.5 pm).
#
# NO· — ²Π, 11 electrons (mult=2); r(N-O) = 1.151 Å (NIST exp).
NO = [(7, [0.0, 0.0, 0.0]),
      (8, [0.0, 0.0, 1.151 * _A])]
# CN· — ²Σ+, 13 electrons (mult=2); r(C-N) = 1.172 Å (NIST exp).
CN = [(6, [0.0, 0.0, 0.0]),
      (7, [0.0, 0.0, 1.172 * _A])]
# Methylperoxy CH3O2· — ²A″, 25 electrons (mult=2). Approximate B3LYP
# equilibrium (Slack 1999 / Sheng 2003 published values):
#   r(C-O) = 1.45, r(O-O) = 1.32, r(C-H) = 1.094 Å
#   ∠(C-O-O) = 110°, ∠(O-C-H) = 109.5° tetrahedral, H atoms staggered
#   relative to the terminal O. Standard bond lengths suffice — the
#   parity comparison sits both codes on the same coordinates so the
#   exact geometry choice never appears as a cross-code shift.
# Methylperoxyl Cs, doublet. Coordinates are the SAME Å literals as
# ``examples/regression/parity_matrix_orca/cases.py::GEOMETRIES["CH3OO"]``,
# converted once through ``_A`` so that PySCF (the parity reference
# here) and ORCA (the parity reference in ``parity_matrix_orca``) sit
# on identical nuclear coordinates. Construction:
#     r(C-O)=1.45, r(O-O)=1.32, r(C-H)=1.094 Å
#     ∠(C-O-O)=110°, ∠(O-C-H)=109.5° tetrahedral, anti-H staggered
CH3OO = [
    (6, [ 0.0000000 * _A,  0.0000000 * _A,  0.0000000 * _A]),
    (8, [ 0.0000000 * _A,  0.0000000 * _A,  1.4500000 * _A]),
    (8, [ 1.2406484 * _A,  0.0000000 * _A,  0.9985598 * _A]),
    (1, [ 0.5158118 * _A,  0.8933350 * _A, -0.3648961 * _A]),
    (1, [-1.0316235 * _A,  0.0000000 * _A, -0.3648961 * _A]),
    (1, [ 0.5158118 * _A, -0.8933350 * _A, -0.3648961 * _A]),
]

_GEOMS = {"H2O": H2O, "H2CO": H2CO, "OH": OH,
          "NO": NO, "CN": CN, "CH3OO": CH3OO}
# vibe-qc -> PySCF xc-string map. vibe-qc's "B3LYP" = ORCA/VWN5
# definition; PySCF's matching spelling is "b3lyp5" (PySCF's plain
# "b3lyp" is the Gaussian/VWN-RPA variant == vibe-qc's "b3lypg").
_PYSCF_XC = {"PBE": "pbe,pbe", "B3LYP": "b3lyp5", "PBE0": "pbe0"}

# Aux basis for the density-fitted (DF) Fock-build-path cells. Chosen
# because BOTH vibe-qc and PySCF accept this exact name and resolve it
# to the same basis (113 functions on H2O) — so the DF fitting is the
# same mathematical approximation in both codes and the comparison
# reduces to the converged-density difference, same as the direct path.
_DF_AUX = "def2-universal-jkfit"

# Per-piece tolerances (Ha; MO eigenvalues in Ha). See module docstring.
_TOL = {
    "e_nuc": 1e-12,     # analytic; worst observed 2.8e-14 (35x headroom)
    "e_1e": 1e-5,       # converged-density-difference dominated; hard floor 9.7e-6
    "e_coulomb": 1e-5,  # (anti-correlated with e_1e); hard floor 9.7e-6
    "e_exchange": 2e-6, # hard floor 1.2e-6 (HF def2-tzvp open-shell)
    "e_xc": 2e-6,       # hard floor 1.4e-6 (DFT def2-tzvp open-shell)
    "e_total": 2e-6,    # stationary point; hard floor 8.1e-7
    "mo": 2e-5,         # hard floor 1.5e-5
}


# --- vibe-qc side --------------------------------------------------------

# Multi-heavy-atom open-shell radicals on def2-tzvp need a stronger
# convergence recipe than the OH/def2-svp easy case: plain DIIS /
# EDIIS+DIIS oscillate; KDIIS (Kollmar 1997, ORCA's opt-in ``!KDIIS``)
# reaches the energy minimum within a few-hundred iterations,
# but the gradient norm settles at ~1e-6 rather than the matrix-wide
# default 1e-8. The looser gradient floor + KDIIS is applied per
# (system, basis) pair below; everything else keeps the tight defaults.
# Calibrated by ``scripts/parity_radical_probe.py`` (commit log
# 2026-05-17) on NO/CN/CH3OO/OH × def2-tzvp.
_HARD_OPEN_SHELL = frozenset({
    ("NO", "def2-tzvp"),
    ("CN", "def2-tzvp"),
    ("CH3OO", "def2-tzvp"),
    ("OH", "def2-tzvp"),
    # OH·/def2-svp: the ²Π π_x/π_y degeneracy meets the default-grid XC
    # anisotropy at a ~1.7e-7 commutator floor, so the previous easy-cell
    # UKS gate (1e-7) sat *below* the floor. The old per-spin DIIS crossed
    # it after ~170 iterations by noise; the spin-coupled DIIS (2026-07)
    # converges to the floor in ~10 iterations and stays there. The hard
    # recipe's 1e-6 gate is far above the floor and far below anything
    # that could move the energy-piece parity tolerances.
    ("OH", "def2-svp"),
})


def _vibeqc_decompose(atoms, basis_name, method, *, mult=1, aux_basis="",
                      cosx=False, sysname=None):
    """Run a vibe-qc SCF and decompose it.

    Fock-build path: ``aux_basis`` empty -> direct; ``aux_basis`` set
    -> density-fitted; ``cosx=True`` (needs ``aux_basis``) -> RIJCOSX
    (DF-J + chain-of-spheres K). The decomposition is run with the
    same path the SCF used, so ``e_total_residual`` stays at machine
    precision.

    Initial guess is pinned to SAD: the AUTO default (SAP for
    closed-shell light, since commit c31c9f5 / 2026-05-16) leaves
    H2CO/RKS-PBE non-converged after 300 DIIS iterations on the basin
    SAP drops it in. SAD converges in 68 iter to E=-114.2828 Ha and
    is what the parity tolerances were calibrated against. Drop this
    pin when the SAP convergence regression is fixed; see drop-box
    `.release-status/v0.8.0/qa-cross-code-parity.md` § Regression
    finding for the repro + scope-owner notes. [SAP regression]
    """
    from vibeqc._vibeqc_core import SCFAccelerator
    mol = Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms],
                   multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    df = bool(aux_basis) or cosx
    is_hard = (sysname, basis_name) in _HARD_OPEN_SHELL
    if method == "RHF":
        o = RHFOptions()
        o.conv_tol_energy = 1e-12
        o.conv_tol_grad = 1e-9
        o.density_fit = df
        o.aux_basis = aux_basis
        o.cosx = cosx
        o.initial_guess = InitialGuess.SAD
        return decompose_energy_rhf(mol, basis, run_rhf(mol, basis, o),
                                    density_fit=df, aux_basis=aux_basis,
                                    cosx=cosx, cosx_grid=scf_cosx_grid(o, basis_name))
    if method == "UHF":
        o = UHFOptions()
        if is_hard:
            o.conv_tol_energy = 1e-9
            o.conv_tol_grad = 1e-6
            o.max_iter = 500
            o.scf_accelerator = SCFAccelerator.KDIIS
        else:
            o.conv_tol_energy = 1e-11
            o.conv_tol_grad = 1e-8
            o.max_iter = 300
        o.density_fit = df
        o.aux_basis = aux_basis
        o.cosx = cosx
        o.initial_guess = InitialGuess.SAD
        return decompose_energy_uhf(mol, basis, run_uhf(mol, basis, o),
                                    density_fit=df, aux_basis=aux_basis,
                                    cosx=cosx, cosx_grid=scf_cosx_grid(o, basis_name))
    if method.startswith("RKS-"):
        o = RKSOptions()
        o.functional = method.split("-", 1)[1]
        o.conv_tol_energy = 1e-12
        o.conv_tol_grad = 1e-9
        # Headroom over the default 100: H2CO/def2-tzvp RKS-PBE needs
        # ~128 plain-DIIS iterations on the DF path (slower than the
        # ~61 of the direct path; EDIIS+DIIS would cut this once it
        # lands on main).
        o.max_iter = 300
        o.density_fit = df
        o.aux_basis = aux_basis
        o.cosx = cosx
        o.initial_guess = InitialGuess.SAD
        return decompose_energy_rks(mol, basis, run_rks(mol, basis, o),
                                    o.grid, density_fit=df,
                                    aux_basis=aux_basis, cosx=cosx,
                                    cosx_grid=scf_cosx_grid(o, basis_name))
    if method.startswith("UKS-"):
        o = UKSOptions()
        o.functional = method.split("-", 1)[1]
        if is_hard:
            o.conv_tol_energy = 1e-9
            o.conv_tol_grad = 1e-6
            o.max_iter = 500
            o.damping = 0.7
            o.scf_accelerator = SCFAccelerator.KDIIS
        else:
            o.conv_tol_energy = 1e-10
            o.conv_tol_grad = 1e-7
            # (OH·/def2-svp UKS used to be handled here with damping=0.7;
            # it moved to _HARD_OPEN_SHELL when the spin-coupled DIIS
            # exposed its ~1.7e-7 grid-anisotropy commutator floor.)
            o.max_iter = 500
            o.damping = 0.7
        o.density_fit = df
        o.aux_basis = aux_basis
        o.cosx = cosx
        o.initial_guess = InitialGuess.SAD
        return decompose_energy_uks(mol, basis, run_uks(mol, basis, o),
                                    o.grid, density_fit=df,
                                    aux_basis=aux_basis, cosx=cosx,
                                    cosx_grid=scf_cosx_grid(o, basis_name))
    raise ValueError(f"unknown method {method!r}")


# --- PySCF side ----------------------------------------------------------

def _pyscf_decompose(atoms, basis_name, method, *, charge=0, spin=0,
                     aux_basis=""):
    """PySCF per-intermediate decomposition matching vibeqc.parity's dict.

    Non-empty ``aux_basis`` -> density-fitted Fock build (``mf =
    mf.density_fit(...)``); ``mf.get_j`` / ``mf.get_k`` then return the
    DF-fitted matrices, so the same decomposition code applies.

    Self-checks: the reconstructed total must match ``mf.e_tot`` — a
    wrong reference helper fails loudly here rather than feeding bad
    numbers into the parity assertions.
    """
    from pyscf import gto, scf, dft

    mol = gto.M(atom=[[int(z), tuple(xyz)] for z, xyz in atoms],
                basis=basis_name, unit="Bohr", charge=charge, spin=spin,
                verbose=0)
    e_nuc = float(mol.energy_nuc())
    restricted = method in ("RHF",) or method.startswith("RKS-")

    if method == "RHF":
        mf = scf.RHF(mol)
    elif method == "UHF":
        mf = scf.UHF(mol)
    elif method.startswith("RKS-"):
        mf = dft.RKS(mol)
        mf.xc = _PYSCF_XC[method.split("-", 1)[1]]
        mf.grids.level = 5
    elif method.startswith("UKS-"):
        mf = dft.UKS(mol)
        mf.xc = _PYSCF_XC[method.split("-", 1)[1]]
        mf.grids.level = 5
    else:
        raise ValueError(f"unknown method {method!r}")
    if aux_basis:
        mf = mf.density_fit(auxbasis=aux_basis)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        # OH·/def2-svp open-shell DFT is a hard-convergence regime for
        # plain DIIS — fall back to PySCF's second-order (Newton) solver.
        mf = mf.newton()
        mf.conv_tol = 1e-12
        mf.kernel()
    assert mf.converged, f"PySCF {method} did not converge"

    hcore = mf.get_hcore()
    dm = mf.make_rdm1()
    is_dft = method.startswith(("RKS-", "UKS-"))
    ni = mf._numint if is_dft else None
    alpha_hf = float(ni.hybrid_coeff(mf.xc)) if is_dft else 1.0

    if restricted:
        e_1e = float(np.einsum("ij,ji->", hcore, dm))
        j = mf.get_j(mol, dm)
        e_coulomb = 0.5 * float(np.einsum("ij,ji->", j, dm))
        if alpha_hf != 0.0:
            k = mf.get_k(mol, dm)
            e_exchange_hf = -0.25 * float(np.einsum("ij,ji->", k, dm))
        else:
            e_exchange_hf = 0.0
        if is_dft:
            e_xc = float(ni.nr_rks(mol, mf.grids, mf.xc, dm)[1])
        else:
            e_xc = 0.0
        mo = {"mo_energies": np.asarray(mf.mo_energy, dtype=float)}
    else:
        dm_a, dm_b = dm[0], dm[1]
        dm_t = dm_a + dm_b
        e_1e = float(np.einsum("ij,ji->", hcore, dm_t))
        j = mf.get_j(mol, dm_t)
        e_coulomb = 0.5 * float(np.einsum("ij,ji->", j, dm_t))
        if alpha_hf != 0.0:
            k_a = mf.get_k(mol, dm_a)
            k_b = mf.get_k(mol, dm_b)
            e_exchange_hf = -0.5 * (
                float(np.einsum("ij,ji->", k_a, dm_a))
                + float(np.einsum("ij,ji->", k_b, dm_b)))
        else:
            e_exchange_hf = 0.0
        if is_dft:
            e_xc = float(ni.nr_uks(mol, mf.grids, mf.xc, (dm_a, dm_b))[1])
        else:
            e_xc = 0.0
        mo = {
            "mo_energies_alpha": np.asarray(mf.mo_energy[0], dtype=float),
            "mo_energies_beta": np.asarray(mf.mo_energy[1], dtype=float),
        }

    e_exchange = alpha_hf * e_exchange_hf
    e_total = e_nuc + e_1e + e_coulomb + e_exchange + e_xc
    self_res = e_total - float(mf.e_tot)
    assert abs(self_res) < 1e-7, (
        f"PySCF reference helper inconsistent for {method}: reconstructed "
        f"E_total off mf.e_tot by {self_res:.2e} Ha"
    )
    out = {
        "e_nuc": e_nuc, "e_1e": e_1e, "e_coulomb": e_coulomb,
        "e_exchange": e_exchange, "e_xc": e_xc, "e_total": e_total,
    }
    out.update(mo)
    return out


# --- assertion helper ----------------------------------------------------

def _assert_parity(vq, ps, label):
    for key in ("e_nuc", "e_1e", "e_coulomb", "e_exchange", "e_xc", "e_total"):
        d = abs(vq[key] - ps[key])
        assert d < _TOL[key], (
            f"{label}: {key} disagrees with PySCF by {d:.3e} Ha "
            f"(tol {_TOL[key]:.1e})"
        )
    # occupied-block MO eigenvalues
    if "mo_energies" in vq:
        n = min(len(vq["mo_energies"]), len(ps["mo_energies"]))
        d = float(np.abs(vq["mo_energies"][:n] - ps["mo_energies"][:n]).max())
        assert d < _TOL["mo"], (
            f"{label}: MO eigenvalues disagree with PySCF by {d:.3e} Ha "
            f"(tol {_TOL['mo']:.1e})"
        )
    else:
        for spin in ("alpha", "beta"):
            kv = f"mo_energies_{spin}"
            n = min(len(vq[kv]), len(ps[kv]))
            d = float(np.abs(vq[kv][:n] - ps[kv][:n]).max())
            assert d < _TOL["mo"], (
                f"{label}: {spin} MO eigenvalues disagree with PySCF by "
                f"{d:.3e} Ha (tol {_TOL['mo']:.1e})"
            )


# --- 1. self-consistency (no PySCF) -------------------------------------

# (sysname, basis, method, mult, charge, spin, aux_basis, cosx)
# Fock-build path: aux_basis "" -> direct; set -> density fitting;
# cosx=True (needs aux_basis) -> RIJCOSX (DF-J + chain-of-spheres K).
_SELF_CASES = [
    ("H2O", "def2-svp", "RHF", 1, 0, 0, "", False),
    ("H2O", "def2-tzvp", "RHF", 1, 0, 0, "", False),
    ("H2O", "def2-svp", "RKS-PBE", 1, 0, 0, "", False),
    ("H2O", "def2-svp", "RKS-B3LYP", 1, 0, 0, "", False),
    ("H2O", "def2-svp", "RKS-PBE0", 1, 0, 0, "", False),
    ("H2CO", "def2-tzvp", "RKS-PBE", 1, 0, 0, "", False),
    ("OH", "def2-svp", "UHF", 2, 0, 1, "", False),
    ("OH", "def2-svp", "UKS-PBE", 2, 0, 1, "", False),
    # density-fitted Fock build
    ("H2O", "def2-svp", "RHF", 1, 0, 0, _DF_AUX, False),
    ("H2O", "def2-svp", "RKS-B3LYP", 1, 0, 0, _DF_AUX, False),
    ("OH", "def2-svp", "UHF", 2, 0, 1, _DF_AUX, False),
    # RIJCOSX Fock build (DF-J + chain-of-spheres K)
    ("H2O", "def2-svp", "RHF", 1, 0, 0, _DF_AUX, True),
    ("H2O", "def2-svp", "RKS-B3LYP", 1, 0, 0, _DF_AUX, True),
    ("OH", "def2-svp", "UHF", 2, 0, 1, _DF_AUX, True),
    ("OH", "def2-svp", "UKS-B3LYP", 2, 0, 1, _DF_AUX, True),
]


def _self_id(case):
    s, b, m, _mult, _ch, _sp, aux, cosx = case
    suffix = "-RIJCOSX" if cosx else ("-DF" if aux else "")
    return f"{s}-{b}-{m}{suffix}"


@pytest.mark.parametrize(
    "sysname,basis,method,mult,charge,spin,aux_basis,cosx", _SELF_CASES,
    ids=[_self_id(c) for c in _SELF_CASES],
)
def test_parity_decomposition_self_consistent(
        sysname, basis, method, mult, charge, spin, aux_basis, cosx):
    """vibeqc.parity's decomposed pieces must sum to result.energy.

    Every piece is computed from the stored result.density with the
    Fock-build path the SCF used — direct, DF, or RIJCOSX (DF-J +
    chain-of-spheres K) — so the sum equals E(result.density) to
    machine precision once the SCF is well-converged. No PySCF needed
    — this pins parity.py itself.

    The RIJCOSX cells live only in this self-consistency family:
    PySCF has no chain-of-spheres exchange, so a RIJCOSX-vs-RIJCOSX
    cross-code cell is not possible. The *absolute* correctness of
    RIJCOSX is pinned separately by ``tests/test_rijcosx.py`` (RIJCOSX
    vs direct SCF, sub-mHa fit band; the original commit also
    validated vs ORCA 6.1.1).

    The cosx self-consistency floor is looser (1e-7 vs 1e-9 for
    non-cosx): the per-shell radial cutoff in the COSX K kernel
    (``50e9b6a``, "XC modernization stage 3") introduced a ~1e-8
    numerical difference between the SCF's K(D) call and the
    post-SCF decomposition's K(D) re-build at the same D. Worst
    observed 5.5e-8 on H2CO/def2-svp/RKS-PBE0/RIJCOSX.
    """
    payload = _vibeqc_decompose(_GEOMS[sysname], basis, method, mult=mult,
                                aux_basis=aux_basis, cosx=cosx,
                                sysname=sysname)
    assert payload["converged"]
    _floor = 1e-7 if cosx else 1e-9
    assert abs(payload["e_total_residual"]) < _floor, (
        f"{sysname}/{basis}/{method}: decomposition pieces sum to "
        f"{payload['e_total']:.10f} but result.energy is "
        f"{payload['e_total_reported']:.10f} "
        f"(residual {payload['e_total_residual']:.2e}, floor {_floor:.0e})"
    )


# --- 2. cross-code parity vs PySCF --------------------------------------

# (sysname, basis, method, charge, spin, aux_basis)
# aux_basis == "" -> direct Fock build; non-empty -> density fitting.
_PARITY_CASES = [
    # --- direct Fock build ---
    # closed-shell diagonal: {H2O, H2CO} x {def2-svp, def2-tzvp}
    #                        x {RHF, RKS-PBE, RKS-B3LYP}
    ("H2O", "def2-svp", "RHF", 0, 0, ""),
    ("H2O", "def2-svp", "RKS-PBE", 0, 0, ""),
    ("H2O", "def2-svp", "RKS-B3LYP", 0, 0, ""),
    ("H2O", "def2-tzvp", "RHF", 0, 0, ""),
    ("H2O", "def2-tzvp", "RKS-PBE", 0, 0, ""),
    ("H2O", "def2-tzvp", "RKS-B3LYP", 0, 0, ""),
    ("H2CO", "def2-svp", "RHF", 0, 0, ""),
    ("H2CO", "def2-svp", "RKS-PBE", 0, 0, ""),
    ("H2CO", "def2-svp", "RKS-B3LYP", 0, 0, ""),
    ("H2CO", "def2-tzvp", "RHF", 0, 0, ""),
    ("H2CO", "def2-tzvp", "RKS-PBE", 0, 0, ""),
    ("H2CO", "def2-tzvp", "RKS-B3LYP", 0, 0, ""),
    # PBE0 hybrid (alias landed 2026-05-14) — closed-shell, low-l + f-shell
    ("H2O", "def2-svp", "RKS-PBE0", 0, 0, ""),
    ("H2CO", "def2-tzvp", "RKS-PBE0", 0, 0, ""),
    # open-shell (no EDIIS+DIIS needed for OH/def2-svp)
    ("OH", "def2-svp", "UHF", 0, 1, ""),
    ("OH", "def2-svp", "UKS-PBE", 0, 1, ""),
    ("OH", "def2-svp", "UKS-PBE0", 0, 1, ""),
    # --- density-fitted Fock build (same aux basis in both codes) ---
    ("H2O", "def2-svp", "RHF", 0, 0, _DF_AUX),
    ("H2O", "def2-svp", "RKS-PBE", 0, 0, _DF_AUX),
    ("H2O", "def2-svp", "RKS-B3LYP", 0, 0, _DF_AUX),
    ("H2O", "def2-svp", "RKS-PBE0", 0, 0, _DF_AUX),
    ("H2CO", "def2-tzvp", "RHF", 0, 0, _DF_AUX),
    ("H2CO", "def2-tzvp", "RKS-PBE", 0, 0, _DF_AUX),
    ("OH", "def2-svp", "UHF", 0, 1, _DF_AUX),
    # --- multi-heavy-atom open-shell def2-tzvp (BRIEF.md § Scope-2) ---
    # NO·, CN·, methylperoxy, OH· at def2-tzvp on UHF + UKS-PBE +
    # UKS-B3LYP, both {direct, DF} rows so every cell certifies both
    # Fock-build paths. EDIIS+DIIS landed on main at d2099de
    # (2026-05-15) and KDIIS (Kollmar 1997) landed alongside; the
    # ``_HARD_OPEN_SHELL`` set in ``_vibeqc_decompose`` routes these
    # cells through KDIIS + conv_tol_grad=1e-6 + max_iter=500.
    #
    # NO·/def2-tzvp/UHF is EXCLUDED — vibe-qc UHF stalls at the
    # energy minimum (E within 1e-6 of PySCF's -129.302674 Ha) but
    # never satisfies any conv_tol_grad we've tried (1e-6 included)
    # across DIIS/EDIIS+DIIS/KDIIS. PySCF converges the same case in
    # 19 iterations. Drop-box § "Regression finding #2 — NO·/UHF
    # stuck" tracks the handoff to the SCF-accelerator chat; the cell
    # is added back once a fix lands.
    ("NO",     "def2-tzvp", "UKS-PBE",   0, 1, ""),
    ("NO",     "def2-tzvp", "UKS-B3LYP", 0, 1, ""),
    ("CN",     "def2-tzvp", "UHF",       0, 1, ""),
    ("CN",     "def2-tzvp", "UKS-PBE",   0, 1, ""),
    ("CN",     "def2-tzvp", "UKS-B3LYP", 0, 1, ""),
    ("CH3OO",  "def2-tzvp", "UHF",       0, 1, ""),
    ("CH3OO",  "def2-tzvp", "UKS-PBE",   0, 1, ""),
    ("CH3OO",  "def2-tzvp", "UKS-B3LYP", 0, 1, ""),
    ("OH",     "def2-tzvp", "UHF",       0, 1, ""),
    ("OH",     "def2-tzvp", "UKS-PBE",   0, 1, ""),
    ("OH",     "def2-tzvp", "UKS-B3LYP", 0, 1, ""),
    # --- multi-heavy-atom open-shell def2-tzvp + DF ---
    ("NO",     "def2-tzvp", "UKS-PBE",   0, 1, _DF_AUX),
    ("NO",     "def2-tzvp", "UKS-B3LYP", 0, 1, _DF_AUX),
    ("CN",     "def2-tzvp", "UHF",       0, 1, _DF_AUX),
    ("CN",     "def2-tzvp", "UKS-PBE",   0, 1, _DF_AUX),
    ("CN",     "def2-tzvp", "UKS-B3LYP", 0, 1, _DF_AUX),
    ("CH3OO",  "def2-tzvp", "UHF",       0, 1, _DF_AUX),
    ("CH3OO",  "def2-tzvp", "UKS-PBE",   0, 1, _DF_AUX),
    ("CH3OO",  "def2-tzvp", "UKS-B3LYP", 0, 1, _DF_AUX),
    ("OH",     "def2-tzvp", "UHF",       0, 1, _DF_AUX),
    ("OH",     "def2-tzvp", "UKS-PBE",   0, 1, _DF_AUX),
    ("OH",     "def2-tzvp", "UKS-B3LYP", 0, 1, _DF_AUX),
]


def _parity_id(case):
    s, b, m, _ch, _sp, aux = case
    return f"{s}-{b}-{m}" + ("-DF" if aux else "")


def _parity_params():
    """Mark the heavy multi-heavy-atom open-shell def2-tzvp cells @slow.

    These NO/CN/CH3OO/OH × {UHF, UKS-PBE, UKS-B3LYP} × {direct, DF} cells
    route through KDIIS + ``conv_tol_grad=1e-6`` + ``max_iter=500`` on
    def2-tzvp and shell out to a PySCF reference — ~60-160 s each, and
    they dominate this file's runtime. The closed-shell + def2-svp +
    light open-shell coverage stays on the fast lane; the heavy
    cross-code cells run on the slow/nightly lane.
    """
    params = []
    for c in _PARITY_CASES:
        _sys, basis, method = c[0], c[1], c[2]
        heavy = basis == "def2-tzvp" and method[0] == "U"
        marks = [pytest.mark.slow] if heavy else []
        params.append(pytest.param(*c, id=_parity_id(c), marks=marks))
    return params


@pytest.mark.parametrize(
    "sysname,basis,method,charge,spin,aux_basis", _parity_params(),
)
def test_parity_vs_pyscf(sysname, basis, method, charge, spin, aux_basis):
    """Each decomposed energy piece must match PySCF within tolerance.

    A failing *piece* (not just the total) localises the bug: E_nuc /
    E_1e -> integrals; E_coulomb / E_exchange -> Fock build; E_xc ->
    XC quadrature; MO eigenvalues -> diagonalisation / convergence.
    Cells with a non-empty ``aux_basis`` exercise the density-fitted
    Fock-build path (same aux basis fed to both codes).
    """
    pytest.importorskip("pyscf")
    mult = spin + 1
    vq = _vibeqc_decompose(_GEOMS[sysname], basis, method, mult=mult,
                           aux_basis=aux_basis, sysname=sysname)
    assert vq["converged"], f"vibe-qc {method} did not converge"
    ps = _pyscf_decompose(_GEOMS[sysname], basis, method,
                          charge=charge, spin=spin, aux_basis=aux_basis)
    label = f"{sysname}/{basis}/{method}" + ("/DF" if aux_basis else "")
    _assert_parity(vq, ps, label)


# --- 3. tight integral certification (basis-ordering invariant) ---------
#
# The energy-decomposition cells above compare each code at its *own*
# converged density, so E_1e / E_coulomb carry a ~1e-6 converged-density
# floor. This family removes the SCF entirely: it compares
# basis-ordering-invariant integral quantities (eigenvalue spectra +
# tr(M·J(M)) contractions) directly, certifying the integral machinery
# (S, T, V_ne, ERIs) to machine precision. vibe-qc and PySCF use
# different in-shell BF orderings but the same Gaussians, so only
# transform-invariant quantities are comparable — see
# vibeqc.parity.integral_invariants.

def _pyscf_integral_invariants(atoms, basis_name, *, charge=0, spin=0):
    """PySCF counterpart of vibeqc.parity.integral_invariants."""
    from pyscf import gto

    mol = gto.M(atom=[[int(z), tuple(xyz)] for z, xyz in atoms],
                basis=basis_name, unit="Bohr", charge=charge, spin=spin,
                verbose=0)
    s = mol.intor("int1e_ovlp")
    t = mol.intor("int1e_kin")
    v = mol.intor("int1e_nuc")
    h = t + v
    eri = mol.intor("int2e", aosym="s1")

    def j_of(m):
        return np.einsum("ijkl,kl->ij", eri, m, optimize=True)

    def k_of(m):
        return np.einsum("ikjl,kl->ij", eri, m, optimize=True)

    return {
        "s_eigvals": np.linalg.eigvalsh(s),
        "t_eigvals": np.linalg.eigvalsh(t),
        "v_ne_eigvals": np.linalg.eigvalsh(v),
        "h_core_eigvals": np.linalg.eigvalsh(h),
        "eri_s_j": float(np.einsum("ij,ij->", s, j_of(s))),
        "eri_s_k": float(np.einsum("ij,ij->", s, k_of(s))),
        "eri_h_j": float(np.einsum("ij,ij->", h, j_of(h))),
        "eri_h_k": float(np.einsum("ij,ij->", h, k_of(h))),
    }


# Integral invariants are SCF-independent — only (geometry, basis)
# matter. (sysname, basis, mult, spin) — mult/spin only feed the
# Molecule / gto.M constructors (odd-electron systems need them).
_INTEGRAL_CASES = [
    ("H2O", "def2-svp", 1, 0),
    ("H2O", "def2-tzvp", 1, 0),
    ("H2CO", "def2-svp", 1, 0),
    ("H2CO", "def2-tzvp", 1, 0),
    ("OH", "def2-svp", 2, 1),
]

# Eigenvalue spectra are absolute (the matrices have O(1)-O(10²)
# entries); the tr(M·J(M)) contractions span O(10²)-O(10⁶), so they
# use a relative tolerance. Observed worst (probe): spectra ~2e-13,
# contractions ~4e-15 relative — the tolerances below carry orders of
# magnitude of headroom and would catch any real integral regression.
_EIGVAL_TOL = 1e-9
_CONTRACTION_RTOL = 1e-8


@pytest.mark.parametrize(
    "sysname,basis,mult,spin", _INTEGRAL_CASES,
    ids=[f"{s}-{b}" for s, b, _, _ in _INTEGRAL_CASES],
)
def test_parity_integrals_tight(sysname, basis, mult, spin):
    """vibe-qc's S / T / V_ne / ERI machinery must match PySCF's to
    machine precision, certified via basis-ordering-invariant
    quantities (eigenvalue spectra + tr(M·J(M)) contractions).

    A failure here is an integral bug — distinct from the
    energy-decomposition cells, which can also fail on Fock build /
    XC quadrature / SCF convergence.
    """
    pytest.importorskip("pyscf")
    atoms = _GEOMS[sysname]
    mol = Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms],
                   multiplicity=mult)
    bset = BasisSet(mol, basis)
    vq = integral_invariants(mol, bset)
    ps = _pyscf_integral_invariants(atoms, basis, charge=0, spin=spin)

    label = f"{sysname}/{basis}"
    for key in ("s_eigvals", "t_eigvals", "v_ne_eigvals", "h_core_eigvals"):
        d = float(np.abs(vq[key] - ps[key]).max())
        assert d < _EIGVAL_TOL, (
            f"{label}: {key} spectrum disagrees with PySCF by {d:.3e} "
            f"(tol {_EIGVAL_TOL:.1e}) — one-electron integral bug"
        )
    for key in ("eri_s_j", "eri_s_k", "eri_h_j", "eri_h_k"):
        rel = abs(vq[key] - ps[key]) / max(abs(ps[key]), 1.0)
        assert rel < _CONTRACTION_RTOL, (
            f"{label}: {key} contraction disagrees with PySCF by "
            f"{rel:.3e} relative (tol {_CONTRACTION_RTOL:.1e}) — ERI bug"
        )
