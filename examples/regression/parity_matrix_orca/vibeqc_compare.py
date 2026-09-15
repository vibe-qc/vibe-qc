"""vibe-qc side of the ORCA parity matrix + the per-piece comparison.

This is the only module in the package that imports vibe-qc. It runs a
vibe-qc SCF for a :class:`~.cases.Cell`, decomposes it with
:mod:`vibeqc.parity`, and compares the result against a parsed ORCA
decomposition.

**The combined two-electron bucket.** ORCA's standard output does not
split the Coulomb and exchange energies (see :mod:`.parse_orca`) — it
prints a single ``Two Electron Energy``. So the comparison sums
vibe-qc's ``e_coulomb + e_exchange`` into one ``e_coulomb_plus_exchange``
bucket and compares that against ORCA's ``e_two_electron - e_xc``. Five
energy buckets (e_nuc, e_1e, e_coulomb_plus_exchange, e_xc, e_total)
plus the MO eigenvalues — still localises a discrepancy to integrals,
the Fock build, or the XC quadrature.

The SCF-options recipe mirrors ``tests/test_parity_hf_dft.py``'s
``_vibeqc_decompose`` (its sibling on the PySCF axis) so the ORCA cells
and the PySCF cells decompose vibe-qc identically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from vibeqc import (
    Atom, BasisSet, InitialGuess, Molecule,
    MP2Options, RHFOptions, RKSOptions, UHFOptions, UKSOptions, UMP2Options,
    run_mp2, run_rhf, run_rks, run_uhf, run_uks, run_ump2,
)
from vibeqc.parity import (
    decompose_energy_rhf, decompose_energy_rks,
    decompose_energy_rmp2, decompose_energy_uhf,
    decompose_energy_uks, decompose_energy_ump2,
    scf_cosx_grid,
)

from .cases import Cell, geometry_bohr

# Aux basis for the density-fitted cells. ORCA's "def2/JK" and
# vibe-qc's "def2-universal-jkfit" resolve to the same Weigend basis,
# so a DF cell is the same approximation on both sides — see
# test_parity_hf_dft.py::_DF_AUX.
_DF_AUX = "def2-universal-jkfit"

# Per-piece tolerances for the vibe-qc-vs-ORCA comparison (Ha; MO
# eigenvalues in Ha), calibrated against the matrix runs (see
# PARITY_REPORT.md). They are **method-aware** — HF/UHF and DFT live in
# genuinely different precision regimes:
#
#   * HF / UHF cells have no XC quadrature. The only cross-code
#     difference is the residual converged-density spread between two
#     well-converged codes — observed ~1e-6 on E_1e / E_coulomb, ~1e-10
#     on E_total, ~7e-7 on the MO eigenvalues. So HF cells are held
#     tight: a real integral or Fock-build regression blows past 1e-5
#     by orders of magnitude.
#
#   * RKS / UKS cells carry the ORCA-grid-vs-vibe-qc-product-grid
#     quadrature difference. ORCA's DefGrid (even at VeryTightSCF) is
#     not vibe-qc's product grid, so the two codes converge to slightly
#     different densities — observed up to ~6.6e-4 on E_1e / E_coulomb
#     (anti-correlated, so ~1.7e-4 in the total), ~2e-4 on E_xc,
#     ~4.3e-4 on the MO eigenvalues. This is a real inter-code
#     difference, NOT a vibe-qc bug. Tighter per-piece DFT
#     certification needs matched integration grids — the cross-code
#     parity handover's standing grid action item, a documented
#     follow-up; running ORCA at a finer grid (DefGrid3) would shrink
#     this band substantially. The DFT bands below carry ~2-3x headroom
#     over the worst observed: a real Fock-build / XC regression (mHa
#     scale or larger — cf. the v0.7.0 Madelung bug) is still caught.
#
# E_nuc is analytic and, on identical Bohr coordinates, matches to
# ~1e-12 regardless of method — held tight for both.
_TOL_HF: Dict[str, float] = {
    "e_nuc": 1e-9,
    "e_1e": 1e-5,
    "e_coulomb_plus_exchange": 1e-5,
    "e_xc": 1e-9,            # HF: exactly 0.0 on both sides
    "e_total": 1e-6,
    "mo": 1e-5,
}
_TOL_DFT: Dict[str, float] = {
    "e_nuc": 1e-9,
    "e_1e": 1.5e-3,                     # converged-density-diff (grid) dominated
    "e_coulomb_plus_exchange": 1.5e-3,  # (anti-correlated with e_1e)
    "e_xc": 5e-4,                       # ORCA grid != vibe-qc product grid
    "e_total": 5e-4,                    # grid diff; mostly cancels vs e_1e/J
    "mo": 1e-3,
}
# RIJCOSX HF/UHF: the chain-of-spheres K kernel adds a *seminumerical*
# fit error on top of the converged-density spread the direct/DF HF
# cells already carry. ``tests/test_rijcosx.py`` documents the
# vibe-qc-internal RIJCOSX-vs-direct gap as "sub-mHa" (worst observed
# 2.8e-4 Ha on UHF/OH). The cross-code OH/UHF RIJCOSX cell here
# observes 3.2e-4 on the total, 1.1e-3 on e_1e (anti-correlated with
# +7.8e-4 on the two-electron bucket), 9.4e-4 max on the MO
# eigenvalues. That's the cross-code COSX gap, not a vibe-qc bug — both
# codes implement the same Neese-2009 algorithm but on different
# COSX grids by default. The band below carries ~2x headroom over the
# worst observed; a real implementation regression (mHa-scale) is
# still caught. Pure-DFT RIJCOSX cells use ``_TOL_DFT`` because the
# COSX flag is a no-op on alpha_HF=0 (cf. test_rijcosx_pure_dft_flag_is_noop);
# hybrid RIJCOSX cells use ``_TOL_DFT`` because the DFT grid quadrature
# gap is the dominant per-piece floor and absorbs the COSX K fit error.
_TOL_RIJCOSX_HF: Dict[str, float] = {
    "e_nuc": 1e-9,
    "e_1e": 2e-3,                       # COSX-K influence on the converged density
    "e_coulomb_plus_exchange": 2e-3,    # COSX K-fit error (sub-mHa)
    "e_xc": 1e-9,                       # HF: exactly 0.0 on both sides
    "e_total": 1e-3,                    # sub-mHa fit band (test_rijcosx.py)
    "mo": 2e-3,
}

# MP2 / UMP2 per-piece band. Data-driven — calibrated 2026-05-18
# against the 6 generated ORCA references (PARITY_REPORT.md). Worst
# observed across {H2O, H2CO, OH} × {def2-svp, cc-pvdz, cc-pvtz} ×
# {canonical, RI}:
#   e_hf     1.26e-10  — both codes run the same canonical 4-index HF
#                        at VeryTightSCF / conv_tol_energy=1e-12, so
#                        the HF reference matches to machine precision.
#                        Band 1e-9 (~8x headroom).
#   e_corr   5.10e-08  — MP2 correlation. Canonical-vs-canonical and
#                        RI-vs-RI both land at ~1-5e-8; the RI cells
#                        are no looser than the canonical ones (both
#                        codes do the same Dunlap fit with the same
#                        aux). Band 2e-7 (~4x headroom).
#   e_total  5.11e-08  — e_hf + e_corr, tracks e_corr. Band 2e-7.
#   e_ss / e_os — channel split. ORCA's `! MP2` / `! RI-MP2` standard
#                 output mode does NOT print the same-spin /
#                 opposite-spin pair-energy decomposition, so these
#                 buckets are currently *not exercised* (the parser
#                 returns None and `_compare_mp2` skips them). Band
#                 1e-6 kept as a sensible default for if a future
#                 cell raises ORCA's print level to surface the split.
# A real MP2-transform / amplitude regression is mHa-scale — orders
# of magnitude past these bands.
_TOL_MP2: Dict[str, float] = {
    "e_hf": 1e-9,
    "e_corr": 2e-7,
    "e_total": 2e-7,
    "e_ss": 1e-6,
    "e_os": 1e-6,
}

# Back-compat / default handle: the DFT (looser) band. Prefer
# ``tol_for_method`` so HF cells get the tight band.
ORCA_TOL: Dict[str, float] = _TOL_DFT


def tol_for_method(method: str, *, cosx: bool = False) -> Dict[str, float]:
    """The per-piece tolerance band for `method` (HF/UHF tight, DFT looser).

    `method` is the vibe-qc decomposition's ``method`` key ("rhf" /
    "uhf" / "rks" / "uks") or a cell method id ("RHF", "RKS-PBE", ...).
    ``cosx`` flags a RIJCOSX cell — HF/UHF cells then route through
    the sub-mHa COSX fit-error band (``_TOL_RIJCOSX_HF``); DFT cells
    keep ``_TOL_DFT`` (the DFT-grid floor dominates the COSX fit
    error, so no separate band is needed).
    """
    m = method.lower()
    if m.startswith(("rhf", "uhf")):
        return _TOL_RIJCOSX_HF if cosx else _TOL_HF
    return _TOL_DFT

# Energy buckets compared per cell, in localisation order: a failing
# bucket points at integrals (e_nuc/e_1e), the Fock build
# (e_coulomb_plus_exchange), or the XC quadrature (e_xc).
_ENERGY_PIECES = (
    "e_nuc", "e_1e", "e_coulomb_plus_exchange", "e_xc", "e_total",
)

# Open-shell radicals in the ORCA matrix that need KDIIS (Kollmar 1997)
# + a looser gradient floor (gtol=1e-6 vs the matrix-wide 1e-7/1e-8).
# Plain DIIS / EDIIS+DIIS can sit on the right energy while oscillating
# around the gradient criterion until max_iter.  OH/def2-svp/UKS-PBE is
# the small-cell representative: current main hits the ORCA-passing
# energy but reports converged=False after 500 iterations unless it uses
# this recipe; with KDIIS it converges in 15 iterations to the same
# energy.  The def2-tzvp multi-heavy rows were the original calibration.
# Mirrors ``tests/test_parity_hf_dft.py::_HARD_OPEN_SHELL``; see that
# file for the calibration trace.
_HARD_OPEN_SHELL = frozenset({
    ("OH", "def2-svp"),
    ("NO", "def2-tzvp"),
    ("CN", "def2-tzvp"),
    ("CH3OO", "def2-tzvp"),
    ("OH", "def2-tzvp"),
})


# vibe-qc RI auxiliary basis for the RI-MP2 cells, keyed on the
# orbital basis. These resolve to the same Weigend-Köhn-Hättig
# correlation-fitting Gaussians ORCA denotes ``<basis>/C`` (see
# orca_input.py::_BASIS_TO_ORCA_CFIT) — so an RI-MP2 cell is the same
# mathematical approximation in both codes. Verified in M5 against the
# generated ORCA references.
_RI_MP2_AUX = {
    "def2-svp": "def2-svp-rifit",
    "def2-tzvp": "def2-tzvp-rifit",
    "cc-pvdz": "cc-pvdz-ri",
    "cc-pvtz": "cc-pvtz-ri",
}


def _vibeqc_decompose_mp2(cell: Cell) -> Dict[str, Any]:
    """Run a vibe-qc MP2 / UMP2 cell and return its parity decomposition.

    Runs the HF reference (RHF / UHF, ``InitialGuess.SAD`` pinned for
    the same reason as the SCF cells) then ``run_mp2`` / ``run_ump2``,
    and decomposes via ``vibeqc.parity.decompose_energy_{r,u}mp2``.
    ``cell.df`` selects canonical four-index MP2 vs RI-MP2.
    """
    atoms = geometry_bohr(cell.system)
    mol = Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms],
                   multiplicity=cell.spin + 1)
    basis = BasisSet(mol, cell.basis)
    df = cell.df
    aux = ""
    if df:
        aux = _RI_MP2_AUX.get(cell.basis, "")
        if not aux:
            raise ValueError(
                f"vibeqc_compare: no RI-MP2 aux mapping for basis "
                f"{cell.basis!r} (cell {cell.cell_id}) — add it to "
                f"_RI_MP2_AUX.")

    method = cell.method.upper()
    if method == "RMP2":
        o = RHFOptions()
        o.conv_tol_energy = 1e-12
        o.conv_tol_grad = 1e-9
        o.initial_guess = InitialGuess.SAD
        rhf = run_rhf(mol, basis, o)
        mp2_opts = MP2Options()
        # Preserve the checked-in ORCA NoFrozenCore comparison protocol.
        mp2_opts.n_frozen_core = 0
        if df:
            mp2_opts.density_fit = True
            mp2_opts.aux_basis = aux
        mp2 = run_mp2(mol, basis, rhf, mp2_opts)
        payload = decompose_energy_rmp2(
            mol, basis, rhf, mp2, density_fit=df, aux_basis=aux)
        payload["converged"] = bool(rhf.converged)
    elif method == "UMP2":
        o = UHFOptions()
        o.conv_tol_energy = 1e-11
        o.conv_tol_grad = 1e-8
        o.max_iter = 300
        o.initial_guess = InitialGuess.SAD
        uhf = run_uhf(mol, basis, o)
        ump2_opts = UMP2Options()
        # Preserve the checked-in ORCA NoFrozenCore comparison protocol.
        ump2_opts.n_frozen_core = 0
        if df:
            ump2_opts.density_fit = True
            ump2_opts.aux_basis = aux
        ump2 = run_ump2(mol, basis, uhf, ump2_opts)
        payload = decompose_energy_ump2(
            mol, basis, uhf, ump2, density_fit=df, aux_basis=aux)
        payload["converged"] = bool(uhf.converged)
    else:
        raise ValueError(f"vibeqc_compare: unknown MP2 method {cell.method!r}")

    payload["cell_id"] = cell.cell_id
    return payload


def vibeqc_decompose(cell: Cell) -> Dict[str, Any]:
    """Run a vibe-qc SCF (or MP2) for `cell` and return its decomposition.

    For an SCF cell the returned dict is
    ``vibeqc.parity.decompose_energy_*``'s payload plus a derived
    ``e_coulomb_plus_exchange`` (= ``e_coulomb + e_exchange``) so it
    lines up with what ORCA exposes. For an MP2 cell
    (``cell.is_mp2``) it is the
    ``vibeqc.parity.decompose_energy_{r,u}mp2`` payload — a different
    shape; :func:`compare` dispatches on ``method``.
    """
    if cell.is_mp2:
        return _vibeqc_decompose_mp2(cell)
    atoms = geometry_bohr(cell.system)
    mol = Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms],
                   multiplicity=cell.spin + 1)
    basis = BasisSet(mol, cell.basis)
    # cosx=True implies the DF-J machinery; ``Cell`` already enforces
    # this (a RIJCOSX cell is constructed with df=True too), but read
    # the effective df flag through ``cell.df or cell.cosx`` so a
    # malformed cell still routes through DF.
    df = cell.df or cell.cosx
    cosx = cell.cosx
    aux = _DF_AUX if df else ""

    # Initial guess pinned to SAD throughout: AUTO (SAP for closed-shell
    # light, since c31c9f5 / 2026-05-16) leaves H2CO/RKS-PBE non-
    # converged after 300 DIIS iterations. SAD is what the parity
    # matrix was calibrated against. Drop the pin once the SAP
    # convergence regression is fixed — drop-box
    # `.release-status/v0.8.0/qa-cross-code-parity.md` § Regression
    # finding tracks it. [SAP regression]
    from vibeqc._vibeqc_core import SCFAccelerator
    is_hard = (cell.system, cell.basis) in _HARD_OPEN_SHELL
    method = cell.method
    if method == "RHF":
        o = RHFOptions()
        o.conv_tol_energy = 1e-12
        o.conv_tol_grad = 1e-9
        o.density_fit = df
        o.aux_basis = aux
        o.cosx = cosx
        o.initial_guess = InitialGuess.SAD
        payload = decompose_energy_rhf(
            mol, basis, run_rhf(mol, basis, o),
            density_fit=df, aux_basis=aux,
            cosx=cosx, cosx_grid=scf_cosx_grid(o, cell.basis) if cosx else None)
    elif method == "UHF":
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
        o.aux_basis = aux
        o.cosx = cosx
        o.initial_guess = InitialGuess.SAD
        payload = decompose_energy_uhf(
            mol, basis, run_uhf(mol, basis, o),
            density_fit=df, aux_basis=aux,
            cosx=cosx, cosx_grid=scf_cosx_grid(o, cell.basis) if cosx else None)
    elif method.startswith("RKS-"):
        o = RKSOptions()
        o.functional = method.split("-", 1)[1]
        o.conv_tol_energy = 1e-12
        o.conv_tol_grad = 1e-9
        o.max_iter = 300
        o.density_fit = df
        o.aux_basis = aux
        o.cosx = cosx
        o.initial_guess = InitialGuess.SAD
        payload = decompose_energy_rks(
            mol, basis, run_rks(mol, basis, o), o.grid,
            density_fit=df, aux_basis=aux,
            cosx=cosx, cosx_grid=scf_cosx_grid(o, cell.basis) if cosx else None)
    elif method.startswith("UKS-"):
        o = UKSOptions()
        o.functional = method.split("-", 1)[1]
        if is_hard:
            o.conv_tol_energy = 1e-9
            o.conv_tol_grad = 1e-6
            o.scf_accelerator = SCFAccelerator.KDIIS
        else:
            o.conv_tol_energy = 1e-10
            o.conv_tol_grad = 1e-7
        o.max_iter = 500
        o.damping = 0.7
        o.density_fit = df
        o.aux_basis = aux
        o.cosx = cosx
        o.initial_guess = InitialGuess.SAD
        payload = decompose_energy_uks(
            mol, basis, run_uks(mol, basis, o), o.grid,
            density_fit=df, aux_basis=aux,
            cosx=cosx, cosx_grid=scf_cosx_grid(o, cell.basis) if cosx else None)
    else:
        raise ValueError(f"vibeqc_compare: unknown method {method!r}")

    payload["e_coulomb_plus_exchange"] = (
        payload["e_coulomb"] + payload["e_exchange"])
    payload["cell_id"] = cell.cell_id
    return payload


def _compare_mp2(
    vq: Dict[str, Any], orca: Dict[str, Any], *,
    tol: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compare an MP2 / UMP2 decomposition against a parsed ORCA one.

    MP2 cells have a different piece set than the SCF cells: ``e_hf``
    (HF reference), ``e_corr`` (correlation), ``e_total``, and — when
    ORCA printed the split — the ``e_ss`` / ``e_os`` channels. No XC,
    no MO comparison (the parity interest is the correlation, and the
    HF reference is already certified by the SCF cells).

    The vibe-qc side carries the channels under ``vq["channels"]``
    (``{e_ss, e_os}`` for RMP2; ``{e_aa, e_bb, e_ab}`` for UMP2 — for
    the cross-code channel comparison the UMP2 same-spin channels are
    summed: ``e_ss = e_aa + e_bb``). ORCA's MP2 output gives a single
    ``e_ss`` / ``e_os`` pair regardless of R/U.

    Returns the same verdict shape as :func:`compare` (``ok`` /
    ``pieces`` / ``mo``) so the report formatter consumes it
    uniformly; ``mo`` is a trivial pass (MP2 cells compare no MOs).
    """
    if tol is None:
        tol = _TOL_MP2

    # vibe-qc channels -> a single (e_ss, e_os) pair, R/U-agnostic.
    ch = vq.get("channels", {})
    if "e_ss" in ch:                       # RMP2
        vq_e_ss, vq_e_os = ch["e_ss"], ch["e_os"]
    else:                                  # UMP2: same-spin = aa + bb
        vq_e_ss = ch.get("e_aa", 0.0) + ch.get("e_bb", 0.0)
        vq_e_os = ch.get("e_ab", 0.0)

    rows: List[Dict[str, Any]] = []
    all_ok = True
    # e_hf / e_corr / e_total always compared.
    for piece in ("e_hf", "e_corr", "e_total"):
        v = float(vq[piece])
        o = float(orca[piece])
        delta = abs(v - o)
        piece_tol = tol[piece]
        ok = delta < piece_tol
        all_ok &= ok
        rows.append({"piece": piece, "vibeqc": v, "orca": o,
                     "delta": delta, "tol": piece_tol, "ok": ok})
    # Channels compared only when ORCA printed the split.
    for piece, vq_val in (("e_ss", vq_e_ss), ("e_os", vq_e_os)):
        o = orca.get(piece)
        if o is None:
            continue
        delta = abs(float(vq_val) - float(o))
        piece_tol = tol[piece]
        ok = delta < piece_tol
        all_ok &= ok
        rows.append({"piece": piece, "vibeqc": float(vq_val),
                     "orca": float(o), "delta": delta,
                     "tol": piece_tol, "ok": ok})

    # MP2 cells compare no MO eigenvalues — trivial-pass the mo slot
    # so the shared report formatter stays uniform.
    mo = {"max_delta": 0.0, "tol": 0.0, "ok": True, "n_compared": 0}
    return {"ok": all_ok, "pieces": rows, "mo": mo}


def compare(
    vq: Dict[str, Any], orca: Dict[str, Any], *,
    tol: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compare a vibe-qc decomposition against a parsed ORCA one.

    Dispatches on ``vq["method"]``: an MP2 / UMP2 decomposition goes
    to :func:`_compare_mp2` (``e_hf`` / ``e_corr`` / ``e_total`` +
    optional channel split, no MO); an SCF decomposition is compared
    below.

    `tol` defaults to the method-and-path-aware band
    (:func:`tol_for_method`, keyed on ``vq["method"]`` and
    ``vq["cosx"]``) — tight for HF/UHF, the sub-mHa COSX fit band
    for HF/UHF RIJCOSX, looser for DFT. Pass an explicit dict to
    override.

    Returns a structured verdict::

        {
          "ok": bool,                         # all pieces within tol
          "pieces": [                         # one row per energy bucket
            {"piece", "vibeqc", "orca", "delta", "tol", "ok"}, ...
          ],
          "mo": {"max_delta", "tol", "ok", "n_compared"},
        }

    The MO comparison mirrors ``test_parity_hf_dft.py::_assert_parity``:
    the leading block both codes printed (ORCA truncates virtuals), per
    spin channel for open-shell cells.
    """
    if str(vq.get("method", "")).lower() in ("rmp2", "ump2"):
        return _compare_mp2(vq, orca, tol=tol)
    if tol is None:
        tol = tol_for_method(str(vq.get("method", "rks")),
                             cosx=bool(vq.get("cosx", False)))
    rows: List[Dict[str, Any]] = []
    all_ok = True
    for piece in _ENERGY_PIECES:
        v = float(vq[piece])
        o = float(orca[piece])
        delta = abs(v - o)
        piece_tol = tol[piece]
        ok = delta < piece_tol
        all_ok &= ok
        rows.append({
            "piece": piece, "vibeqc": v, "orca": o,
            "delta": delta, "tol": piece_tol, "ok": ok,
        })

    mo_tol = tol["mo"]
    if "mo_energies" in vq and "mo_energies" in orca:
        n = min(len(vq["mo_energies"]), len(orca["mo_energies"]))
        max_d = float(np.abs(
            np.asarray(vq["mo_energies"][:n], dtype=float)
            - np.asarray(orca["mo_energies"][:n], dtype=float)
        ).max()) if n else 0.0
        mo = {"max_delta": max_d, "tol": mo_tol, "ok": max_d < mo_tol,
              "n_compared": n}
    else:
        # open shell: both spin channels
        worst = 0.0
        total_n = 0
        for spin in ("alpha", "beta"):
            kv = f"mo_energies_{spin}"
            n = min(len(vq[kv]), len(orca[kv]))
            total_n += n
            if n:
                d = float(np.abs(
                    np.asarray(vq[kv][:n], dtype=float)
                    - np.asarray(orca[kv][:n], dtype=float)
                ).max())
                worst = max(worst, d)
        mo = {"max_delta": worst, "tol": mo_tol, "ok": worst < mo_tol,
              "n_compared": total_n}
    all_ok &= mo["ok"]

    return {"ok": all_ok, "pieces": rows, "mo": mo}
