"""Per-intermediate energy decomposition for cross-code HF/DFT parity.

The total SCF energy alone is a coarse correctness check -- when it
disagrees with PySCF or ORCA, it does not say *which* piece is wrong.
This module decomposes a converged vibe-qc SCF result into the same
named intermediates every quantum-chemistry code prints:

    E_total = E_nuc + E_1e + E_coulomb + E_exchange + E_xc

so a parity test can assert each piece independently and localise a
discrepancy to integrals (E_nuc, E_1e), the Fock build (E_coulomb,
E_exchange), or the XC quadrature (E_xc).

This is the vibe-qc *side* of the v0.13.5 HF/DFT certification matrix
(see ``handovers/HANDOVER_CROSS_CODE_PARITY.md``). It is deliberately
self-contained -- it imports nothing from PySCF / ORCA / any other
quantum-chemistry program (CLAUDE.md Sec. 10). The reference side runs
out-of-process; the comparison lives in ``tests/test_parity_hf_dft.py``.

**Every piece is computed from the *stored* ``result.density``** -- not
from the SCF's cached per-iteration energies -- so the decomposition is
internally self-consistent: the pieces sum to ``E(result.density)``,
which equals ``result.energy`` to SCF-stationarity precision (~1e-9 or
tighter for a well-converged result). This matters because the
RKS/UKS result object caches ``e_coulomb`` / ``e_xc`` from the *input*
density of the final SCF iteration, which differs from the stored
output ``result.density`` by ~1e-7 on a loosely-converged run; mixing
those would corrupt a 1e-9-level E_J parity assertion.

**Fock-build path.** Three paths, matching the SCF drivers -- pick the
one the SCF actually used so the decomposition stays self-consistent:
``density_fit=False`` (default) uses the exact 4-index ERI tensor;
``density_fit=True`` (+ ``aux_basis``) uses the DF-fitted J / K;
``cosx=True`` (RIJCOSX -- also needs ``density_fit`` + ``aux_basis``)
uses DF-J + the seminumerical chain-of-spheres K. The XC quadrature is
unaffected by the J / K path.

Conventions (verified to ~1e-10 Ha against the SCF on tight runs):

* ``e_nuc``       -- ``Molecule.nuclear_repulsion()``.
* ``e_1e``        -- tr(D_tot . H_core), H_core = T + V_ne.
* ``e_coulomb``   -- 1/2 tr(D_tot . J(D_tot)), classical Coulomb.
* ``e_exchange_hf`` -- the *full, unscaled* HF exchange energy:
  RHF/RKS  -1/4 tr(D . K(D));  UHF/UKS  -1/2 S_s tr(D_s . K(D_s)).
* ``alpha_hf``    -- HF-exchange fraction (HF: 1.0; KS: from the
  functional -- 0.0 for pure DFT, e.g. 0.2 for B3LYP).
* ``e_exchange``  -- a_HF . e_exchange_hf, the term that enters E_total.
* ``e_xc``        -- DFT XC energy S_g w_g e_xc(g) (HF: 0.0), evaluated
  on ``grid_options`` (default: ``GridOptions()``); pass the same grid
  the SCF used for the closest match to ``result.e_xc``.

``compute_eri`` materialises the full (n_bf⁴) ERI tensor -- fine for the
small molecules in the parity matrix (<= ~100 basis functions), not a
production Fock path.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from . import _vibeqc_core as _core
from ._vibeqc_core import BasisSet, Functional, GridOptions, XCKind
from .density_fitting import DensityFitting


__all__ = [
    "nuclear_repulsion_energy",
    "integral_invariants",
    "scf_cosx_grid",
    "decompose_energy_rhf",
    "decompose_energy_uhf",
    "decompose_energy_rks",
    "decompose_energy_uks",
    "decompose_energy_rmp2",
    "decompose_energy_ump2",
]

def scf_cosx_grid(options, basis_name: str):
    """The COSX integration grid an SCF with these ``options`` actually
    builds. Pass it as ``cosx_grid`` to ``decompose_energy_*`` so the
    decomposition rebuilds K on the **same** grid the SCF used (else the
    self-consistency check fails once ``cosx_grid_level`` selects a GridX
    tier).

    Mirrors the C++ driver gate ``cosx_use_gridx(grid_level, conv_tol_grad)``
    (cosx.hpp): ``grid_level == 0`` keeps the supplied legacy
    ``options.cosx_grid``; ``>= 1`` resolves a GridX tier at any tolerance;
    AUTO (``-1``, the default) resolves a GridX tier only when
    ``conv_tol_grad >= 1e-6`` (the GridX commutator floor) and otherwise
    falls back to the legacy grid, exactly as the SCF does, so a tight-tol
    SCF that fell back to the legacy grid is decomposed on the legacy grid.

    Note: the decomposition rebuilds K with ``compute_cosx_k`` (no one-centre
    correction), so on a GridX grid (sparse core) the decomposition omits the
    SCF's one-centre exchange correction; ``decompose_energy_*`` is therefore
    exactly self-consistent only on the legacy grid. Tight-tol SCFs (which
    auto-fall-back to legacy) decompose exactly; normal-tol GridX runs carry
    the one-centre residual.
    """
    if not getattr(options, "cosx", False):
        return options.cosx_grid
    grid_level = getattr(options, "cosx_grid_level", 0)
    if grid_level == 0:
        return options.cosx_grid
    # AUTO (-1): mirror the conv-tol fallback the C++ driver applies.
    if grid_level <= -1:
        conv = float(getattr(options, "conv_tol_grad", 1e-6) or 1e-6)
        if conv < 1e-6:
            return options.cosx_grid          # auto + tight -> legacy grid
    from . import (cosx_grid_options_for_level, resolve_cosx_grid_level,
                   cosx_basis_cardinality_from_name)
    card = cosx_basis_cardinality_from_name(basis_name)
    return cosx_grid_options_for_level(
        resolve_cosx_grid_level(grid_level, card))


# A J / K builder pair: each maps a density matrix to its J / K matrix.
_JKBuilders = Tuple[Callable[[np.ndarray], np.ndarray],
                    Callable[[np.ndarray], np.ndarray]]


def nuclear_repulsion_energy(mol) -> float:
    """S_{A<B} Z_A Z_B / |R_A - R_B| in Hartree.

    Thin wrapper over the canonical ``Molecule.nuclear_repulsion()`` --
    exposed here so the parity decomposition has a single,
    self-documenting accessor for every named energy piece.
    """
    return float(mol.nuclear_repulsion())


def integral_invariants(mol, basis) -> Dict[str, Any]:
    """Basis-ordering-invariant integral quantities for cross-code parity.

    vibe-qc and PySCF resolve the same basis-set names to the same
    Gaussians, but order the basis functions within a shell
    differently -- so the raw S / T / V_ne / ERI matrices are related
    by an (unknown) orthogonal transform ``P`` and cannot be compared
    element-wise. Quantities that are *invariant* under that transform
    can, and they certify the integral machinery to machine precision
    without ever needing the permutation map:

    * **Eigenvalue spectra** of S, T, V_ne, H_core -- invariant under
      any orthogonal similarity ``M -> P M Pᵀ``.
    * **Full contractions** ``tr(M . J(M))`` / ``tr(M . K(M))`` for a
      symmetric test matrix ``M`` -- invariant because ``J``/``K`` are
      built from the ERI tensor, and ``tr(M . J(M))`` with
      ``M -> P M Pᵀ`` and the 4-index-transformed ERI is unchanged.
      ``S`` and ``H_core`` are used as the (code-local but
      transform-covariant) test matrices.

    Returns a dict of: ``s_eigvals``, ``t_eigvals``, ``v_ne_eigvals``,
    ``h_core_eigvals`` (sorted ndarrays) and ``eri_s_j``, ``eri_s_k``,
    ``eri_h_j``, ``eri_h_k`` (floats). ``compute_eri`` materialises the
    full n_bf⁴ tensor -- keep to <= ~100 basis functions.
    """
    s = np.asarray(_core.compute_overlap(basis))
    t = np.asarray(_core.compute_kinetic(basis))
    v = np.asarray(_core.compute_nuclear(basis, mol))
    h = t + v
    eri = np.asarray(_core.compute_eri(basis))

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


def _hcore(basis, mol) -> np.ndarray:
    """H_core = T + V_ne."""
    t = np.asarray(_core.compute_kinetic(basis))
    v = np.asarray(_core.compute_nuclear(basis, mol))
    return t + v


# COSX integration grid tier -- mirrors the C++
# ``default_cosx_grid_options()`` (cosx.hpp): ~1/8 the XC default.
_COSX_GRID_RADIAL, _COSX_GRID_THETA, _COSX_GRID_PHI = 35, 9, 18


def _make_jk_builders(
    mol, basis, density_fit: bool, aux_basis: str,
    cosx: bool = False, cosx_grid: Optional[GridOptions] = None,
    scf_mode: Optional["_core.SCFMode"] = None,
    scf_mode_auto_threshold: int = 140,
    schwarz_threshold: float = 1e-10,
) -> _JKBuilders:
    """Build the (J, K) matrix-builder closures for the requested path.

    Four Fock-build paths, matching the SCF drivers:

    * ``cosx=True`` -> **RIJCOSX**: J from DF, K from the seminumerical
      chain-of-spheres kernel (``compute_cosx_k``). Requires
      ``density_fit=True`` + ``aux_basis`` (RIJCOSX = RI-J + COSX-K).
      ``cosx_grid`` (a ``GridOptions``) defaults to the COSX tier;
      pass the grid the SCF used for a self-consistent decomposition.
    * ``density_fit=True`` -> DF-fitted J / K via :class:`DensityFitting`.
    * ``scf_mode=SCFMode.DIRECT`` (or ``AUTO`` + n_bf > threshold) ->
      on-the-fly Schwarz-screened libint quartet evaluation via
      :func:`vibeqc.make_direct_jk_builder`. The path that survives at
      ~50-atom / def2-SVP scale where the in-core ERI tensor OOMs.
    * neither -> exact 4-index ERI contraction. OOM-bound at ~250 BF.

    ``scf_mode`` defaults to ``CONVENTIONAL`` for back-compat (the
    historical decomposition path); pass ``DIRECT`` or ``AUTO`` to
    decompose a result from the direct SCF path without rebuilding
    the n⁴ tensor.
    """
    if cosx:
        if not density_fit or not aux_basis:
            raise ValueError(
                "decompose_energy_*: cosx=True is RIJCOSX -- requires "
                "density_fit=True and aux_basis (RI-J + COSX-K)")
        df = DensityFitting(
            basis,
            BasisSet(mol, aux_basis),
            aux_basis_name=aux_basis,
            molecule=mol,
        )
        if cosx_grid is None:
            cosx_grid = GridOptions()
            cosx_grid.n_radial = _COSX_GRID_RADIAL
            cosx_grid.n_theta = _COSX_GRID_THETA
            cosx_grid.n_phi = _COSX_GRID_PHI
        grid = _core.build_grid(mol, cosx_grid)
        return (
            lambda d: np.asarray(df.build_J(d)),
            lambda d: np.asarray(_core.compute_cosx_k(basis, d, grid)),
        )
    if density_fit:
        if not aux_basis:
            raise ValueError(
                "decompose_energy_*: density_fit=True requires aux_basis")
        df = DensityFitting(
            basis,
            BasisSet(mol, aux_basis),
            aux_basis_name=aux_basis,
            molecule=mol,
        )
        return (
            lambda d: np.asarray(df.build_J(d)),
            lambda d: np.asarray(df.build_K_density(d)),
        )
    # Non-DF, non-COSX. Honour scf_mode (default CONVENTIONAL for
    # back-compat) -- direct mode dodges the n⁴ tensor build for
    # large systems.
    mode = scf_mode if scf_mode is not None else _core.SCFMode.CONVENTIONAL
    if mode == _core.SCFMode.AUTO:
        mode = (_core.SCFMode.DIRECT
                if basis.nbasis > scf_mode_auto_threshold
                else _core.SCFMode.CONVENTIONAL)
    if mode == _core.SCFMode.DIRECT:
        jk = _core.make_direct_jk_builder(basis, schwarz_threshold)
        return (
            lambda d: np.asarray(jk.build_J(d)),
            lambda d: np.asarray(jk.build_K(d)),
        )
    eri = np.asarray(_core.compute_eri(basis))
    # J_muν = S_ls (muν|ls) D_ls ; K_muν = S_ls (mul|νs) D_ls  (chemist).
    return (
        lambda d: np.einsum("ijkl,kl->ij", eri, d, optimize=True),
        lambda d: np.einsum("ikjl,kl->ij", eri, d, optimize=True),
    )


# --------------------------------------------------------------------------
# XC energy from a stored density (so the decomposition is self-consistent
# with result.density rather than the SCF's per-iteration cache).
# --------------------------------------------------------------------------

def _density_on_grid(chi, dchi, d):
    """r and (for GGA) gradr on the grid from AO values and a density.

    ``dchi`` is ``(gx, gy, gz)`` or ``None`` for the LDA path.
    Returns ``(rho, (grx, gry, grz) | None)``.
    """
    chi_d = chi @ d
    rho = np.einsum("gi,gi->g", chi_d, chi)
    if dchi is None:
        return rho, None
    gx, gy, gz = dchi
    grx = 2.0 * np.einsum("gi,gi->g", chi_d, gx)
    gry = 2.0 * np.einsum("gi,gi->g", chi_d, gy)
    grz = 2.0 * np.einsum("gi,gi->g", chi_d, gz)
    return rho, (grx, gry, grz)


def _xc_energy_restricted(mol, basis, density, functional, grid_options):
    """E_xc = S_g w_g e_xc(g) for a closed-shell density."""
    grid = _core.build_grid(mol, grid_options)
    pts = np.asarray(grid.points)
    w = np.asarray(grid.weights)
    is_gga = functional.kind == XCKind.GGA
    if is_gga:
        chi, gx, gy, gz = _core.evaluate_ao_with_gradient(basis, pts)
        rho, grad = _density_on_grid(np.asarray(chi),
                                     (np.asarray(gx), np.asarray(gy),
                                      np.asarray(gz)), density)
        sigma = grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2
    else:
        chi = np.asarray(_core.evaluate_ao(basis, pts))
        rho, _ = _density_on_grid(chi, None, density)
        sigma = np.zeros(0, dtype=float)
    exc, _v_rho, _v_sigma = functional.eval_unpolarised(rho, sigma)
    return float(np.sum(w * np.asarray(exc)))


def _xc_energy_unrestricted(mol, basis, density_a, density_b,
                            functional_name, grid_options):
    """E_xc = S_g w_g e_xc(g) for an open-shell (a, b) density pair."""
    func = Functional(functional_name, spin=2)
    grid = _core.build_grid(mol, grid_options)
    pts = np.asarray(grid.points)
    w = np.asarray(grid.weights)
    is_gga = func.kind == XCKind.GGA
    if is_gga:
        chi, gx, gy, gz = _core.evaluate_ao_with_gradient(basis, pts)
        chi = np.asarray(chi)
        dchi = (np.asarray(gx), np.asarray(gy), np.asarray(gz))
        rho_a, grad_a = _density_on_grid(chi, dchi, density_a)
        rho_b, grad_b = _density_on_grid(chi, dchi, density_b)
        sigma_aa = grad_a[0] ** 2 + grad_a[1] ** 2 + grad_a[2] ** 2
        sigma_bb = grad_b[0] ** 2 + grad_b[1] ** 2 + grad_b[2] ** 2
        sigma_ab = (grad_a[0] * grad_b[0] + grad_a[1] * grad_b[1]
                    + grad_a[2] * grad_b[2])
    else:
        chi = np.asarray(_core.evaluate_ao(basis, pts))
        rho_a, _ = _density_on_grid(chi, None, density_a)
        rho_b, _ = _density_on_grid(chi, None, density_b)
        sigma_aa = sigma_ab = sigma_bb = np.zeros(0, dtype=float)
    exc = func.eval_polarised(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb)[0]
    return float(np.sum(w * np.asarray(exc)))


# --------------------------------------------------------------------------
# Decomposition payloads.
# --------------------------------------------------------------------------

def _restricted_payload(
    method: str, mol, basis, result, *,
    alpha_hf: float, e_xc: float, jk: _JKBuilders,
    density_fit: bool, aux_basis: str, cosx: bool = False,
) -> Dict[str, Any]:
    """Shared decomposition for the closed-shell (RHF / RKS) path."""
    d = np.asarray(result.density)
    h = _hcore(basis, mol)
    j_build, k_build = jk

    e_nuc = nuclear_repulsion_energy(mol)
    e_1e = float(np.einsum("ij,ij->", d, h))
    e_coulomb = 0.5 * float(np.einsum("ij,ij->", d, j_build(d)))
    e_exchange_hf = -0.25 * float(np.einsum("ij,ij->", d, k_build(d)))
    e_exchange = alpha_hf * e_exchange_hf

    e_total = e_nuc + e_1e + e_coulomb + e_exchange + e_xc
    return {
        "method": method,
        "density_fit": density_fit,
        "aux_basis": aux_basis if density_fit else "",
        "cosx": cosx,
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "e_nuc": e_nuc,
        "e_1e": e_1e,
        "e_coulomb": e_coulomb,
        "e_exchange_hf": e_exchange_hf,
        "alpha_hf": float(alpha_hf),
        "e_exchange": e_exchange,
        "e_xc": float(e_xc),
        "e_total": e_total,
        "e_total_reported": float(result.energy),
        "e_total_residual": e_total - float(result.energy),
        "mo_energies": np.asarray(result.mo_energies, dtype=float),
    }


def _unrestricted_payload(
    method: str, mol, basis, result, *,
    alpha_hf: float, e_xc: float, jk: _JKBuilders,
    density_fit: bool, aux_basis: str, cosx: bool = False,
) -> Dict[str, Any]:
    """Shared decomposition for the open-shell (UHF / UKS) path."""
    da = np.asarray(result.density_alpha)
    db = np.asarray(result.density_beta)
    dt = da + db
    h = _hcore(basis, mol)
    j_build, k_build = jk

    e_nuc = nuclear_repulsion_energy(mol)
    e_1e = float(np.einsum("ij,ij->", dt, h))
    e_coulomb = 0.5 * float(np.einsum("ij,ij->", dt, j_build(dt)))
    e_exchange_hf = -0.5 * (
        float(np.einsum("ij,ij->", da, k_build(da)))
        + float(np.einsum("ij,ij->", db, k_build(db)))
    )
    e_exchange = alpha_hf * e_exchange_hf

    e_total = e_nuc + e_1e + e_coulomb + e_exchange + e_xc
    return {
        "method": method,
        "density_fit": density_fit,
        "aux_basis": aux_basis if density_fit else "",
        "cosx": cosx,
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "e_nuc": e_nuc,
        "e_1e": e_1e,
        "e_coulomb": e_coulomb,
        "e_exchange_hf": e_exchange_hf,
        "alpha_hf": float(alpha_hf),
        "e_exchange": e_exchange,
        "e_xc": float(e_xc),
        "e_total": e_total,
        "e_total_reported": float(result.energy),
        "e_total_residual": e_total - float(result.energy),
        "mo_energies_alpha": np.asarray(result.mo_energies_alpha, dtype=float),
        "mo_energies_beta": np.asarray(result.mo_energies_beta, dtype=float),
    }


def decompose_energy_rhf(
    mol, basis, result, *,
    density_fit: bool = False, aux_basis: str = "",
    cosx: bool = False, cosx_grid: Optional[GridOptions] = None,
) -> Dict[str, Any]:
    """Decompose a converged ``RHFResult`` into named energy pieces.

    Returns a dict with keys ``e_nuc``, ``e_1e``, ``e_coulomb``,
    ``e_exchange_hf``, ``alpha_hf`` (= 1.0), ``e_exchange``,
    ``e_xc`` (= 0.0), ``e_total``, ``e_total_reported``,
    ``e_total_residual``, ``mo_energies``, ``converged``, ``n_iter``,
    ``density_fit``, ``aux_basis``, ``cosx``.

    Fock-build path:
      * ``density_fit=True`` + ``aux_basis`` -> decompose a DF-SCF
        result with the DF-fitted J / K.
      * ``cosx=True`` (also needs ``density_fit`` + ``aux_basis``) ->
        decompose a **RIJCOSX** result: J from DF, K from the
        seminumerical chain-of-spheres kernel. Pass ``cosx_grid``
        (the SCF's ``cosx_grid``) for a self-consistent decomposition.

    Decomposing with the path the SCF actually used keeps
    ``e_total_residual`` at machine precision.
    """
    jk = _make_jk_builders(mol, basis, density_fit, aux_basis,
                           cosx=cosx, cosx_grid=cosx_grid)
    return _restricted_payload(
        "rhf", mol, basis, result, alpha_hf=1.0, e_xc=0.0, jk=jk,
        density_fit=density_fit, aux_basis=aux_basis, cosx=cosx)


def decompose_energy_uhf(
    mol, basis, result, *,
    density_fit: bool = False, aux_basis: str = "",
    cosx: bool = False, cosx_grid: Optional[GridOptions] = None,
) -> Dict[str, Any]:
    """Decompose a converged ``UHFResult`` into named energy pieces.

    Same dict shape as :func:`decompose_energy_rhf` but with
    ``mo_energies_alpha`` / ``mo_energies_beta`` instead of
    ``mo_energies``. Supports the DF and RIJCOSX (``cosx=True``)
    Fock-build paths -- see :func:`decompose_energy_rhf`.
    """
    jk = _make_jk_builders(mol, basis, density_fit, aux_basis,
                           cosx=cosx, cosx_grid=cosx_grid)
    return _unrestricted_payload(
        "uhf", mol, basis, result, alpha_hf=1.0, e_xc=0.0, jk=jk,
        density_fit=density_fit, aux_basis=aux_basis, cosx=cosx)


def decompose_energy_rks(
    mol, basis, result, grid_options: Optional[GridOptions] = None, *,
    density_fit: bool = False, aux_basis: str = "",
    cosx: bool = False, cosx_grid: Optional[GridOptions] = None,
) -> Dict[str, Any]:
    """Decompose a converged ``RKSResult`` into named energy pieces.

    ``e_xc`` is recomputed from ``result.density`` on ``grid_options``
    (default ``GridOptions()``) so the whole decomposition is
    self-consistent with the stored density. Pass the grid the SCF
    used for the tightest match to ``result.e_xc``. ``alpha_hf`` is
    read from the functional. The J / K Fock-build path (direct / DF /
    RIJCOSX -- see :func:`decompose_energy_rhf`) affects only the J / K
    pieces, not the XC quadrature; for ``cosx=True`` pass the SCF's
    ``cosx_grid``.
    """
    if grid_options is None:
        grid_options = GridOptions()
    func = Functional(result.functional)
    e_xc = _xc_energy_restricted(
        mol, basis, np.asarray(result.density), func, grid_options)
    jk = _make_jk_builders(mol, basis, density_fit, aux_basis,
                           cosx=cosx, cosx_grid=cosx_grid)
    return _restricted_payload(
        "rks", mol, basis, result,
        alpha_hf=float(func.hf_exchange_fraction), e_xc=e_xc, jk=jk,
        density_fit=density_fit, aux_basis=aux_basis, cosx=cosx)


def decompose_energy_uks(
    mol, basis, result, grid_options: Optional[GridOptions] = None, *,
    density_fit: bool = False, aux_basis: str = "",
    cosx: bool = False, cosx_grid: Optional[GridOptions] = None,
) -> Dict[str, Any]:
    """Decompose a converged ``UKSResult`` into named energy pieces.

    Open-shell counterpart of :func:`decompose_energy_rks`.
    """
    if grid_options is None:
        grid_options = GridOptions()
    func = Functional(result.functional)
    e_xc = _xc_energy_unrestricted(
        mol, basis, np.asarray(result.density_alpha),
        np.asarray(result.density_beta), result.functional, grid_options)
    jk = _make_jk_builders(mol, basis, density_fit, aux_basis,
                           cosx=cosx, cosx_grid=cosx_grid)
    return _unrestricted_payload(
        "uks", mol, basis, result,
        alpha_hf=float(func.hf_exchange_fraction), e_xc=e_xc, jk=jk,
        density_fit=density_fit, aux_basis=aux_basis, cosx=cosx)


# --------------------------------------------------------------------------
# Per-intermediate MP2 decomposition (vibe-qc side).
#
# MP2 is a single-shot post-HF correction: a converged HF reference + an
# MP2 correlation correction with a same-spin (aa + bb) / opposite-spin
# (ab + ba) decomposition. The decomposition surface mirrors the
# HF/DFT helpers:
#
#   e_hf            -- HF (or UHF) reference total. Pulled from the
#                     stored RHF / UHF result so a *single* SCF run
#                     drives both the HF parity check and the MP2
#                     parity check (the post-HF cross-code work the
#                     mp2_benchmarks suite at
#                     ``examples/molecular/mp2_benchmarks`` builds on).
#   e_corr          -- MP2 correlation energy as reported by
#                     ``MP2Result.e_correlation`` (or ``UMP2Result``).
#   e_ss / e_os     -- Same-spin / opposite-spin channels. For RMP2
#                     ``e_ss + e_os ≈ e_corr`` to machine precision
#                     (the *unscaled* canonical-MP2 case). For
#                     SCS-MP2 / SOS-MP2 the result objects store
#                     scaled correlation in ``e_correlation`` while
#                     ``e_ss`` / ``e_os`` remain the unscaled channel
#                     totals -- the SS+OS residual exposes the scaling.
#   e_total         -- ``e_hf + e_corr``. Matches ORCA's MP2 total.
#
# The decomposition does **not** re-evaluate any integrals; it threads
# the already-computed pieces into a single dict so the cross-code
# comparator can assert each independently and localise a discrepancy
# to the SCF reference vs the post-HF AO->MO transform vs the MP2
# amplitude / channel decomposition.
# --------------------------------------------------------------------------


def _mp2_payload(
    method: str, hf_decomp: Dict[str, Any], mp2_result, *,
    density_fit: bool, aux_basis: str,
    ss_label: str = "e_ss", os_label: str = "e_os",
    channel_labels: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Shared MP2 / UMP2 decomposition payload.

    ``channel_labels`` is ``("e_ss", "e_os")`` for RMP2 and
    ``("e_aa", "e_bb", "e_ab")`` for UMP2; the corresponding
    ``MP2Result`` / ``UMP2Result`` attributes are pulled by name.
    """
    if channel_labels is None:
        channel_labels = (ss_label, os_label)

    e_hf = float(hf_decomp["e_total"])
    e_corr = float(mp2_result.e_correlation)
    e_total = e_hf + e_corr
    channels = {name: float(getattr(mp2_result, name))
                for name in channel_labels}
    channel_sum = sum(channels.values())
    return {
        "method": method,
        "density_fit": bool(density_fit),
        "aux_basis": aux_basis if density_fit else "",
        "e_hf": e_hf,
        "e_corr": e_corr,
        "e_total": e_total,
        "e_total_reported": float(mp2_result.e_total),
        "e_total_residual": e_total - float(mp2_result.e_total),
        # Channel decomposition + an SS+OS-vs-e_corr residual. For
        # canonical (R/U)MP2 the residual is ~1e-14 (machine precision);
        # for SCS / SOS the residual surfaces the c_ss/c_os scaling
        # applied to e_correlation.
        "channels": channels,
        "e_channel_residual": channel_sum - e_corr,
        "hf_decomp": hf_decomp,
    }


def decompose_energy_rmp2(
    mol, basis, rhf_result, mp2_result, *,
    density_fit: bool = False, aux_basis: str = "",
) -> Dict[str, Any]:
    """Decompose a converged ``MP2Result`` into named energy pieces.

    The HF reference is decomposed via :func:`decompose_energy_rhf`
    (always direct -- the HF parity check stands on its own). The MP2
    correction is layered on top: ``e_corr``, the ``(e_ss, e_os)``
    channel split, and an ``e_total = e_hf + e_corr`` reconstruction.

    Returns a dict with keys:

    * ``method``       -- ``"rmp2"``.
    * ``density_fit``  -- whether ``run_mp2`` was driven with DF (RI-MP2).
    * ``aux_basis``    -- the RI auxiliary basis (e.g. ``"cc-pvtz-ri"``).
    * ``e_hf``, ``e_corr``, ``e_total``, ``e_total_reported``,
      ``e_total_residual``.
    * ``channels``     -- ``{"e_ss": ..., "e_os": ...}`` (unscaled
                         canonical channels).
    * ``e_channel_residual`` -- ``e_ss + e_os - e_corr`` (machine
      precision for canonical MP2; non-zero for SCS / SOS scaled
      ``e_correlation``).
    * ``hf_decomp``    -- the nested HF dict from
      :func:`decompose_energy_rhf`.

    ``rhf_result`` should be the SCF result the MP2 was driven from --
    same molecule, basis, density. Mismatched references would surface
    in ``e_total_residual`` because the MP2 ``e_total`` stores the
    SCF's own e_hf and the decomposition reconstructs ``e_hf + e_corr``
    from the passed RHF.
    """
    hf_decomp = decompose_energy_rhf(mol, basis, rhf_result, density_fit=False)
    return _mp2_payload(
        "rmp2", hf_decomp, mp2_result,
        density_fit=density_fit, aux_basis=aux_basis,
        channel_labels=("e_ss", "e_os"))


def decompose_energy_ump2(
    mol, basis, uhf_result, ump2_result, *,
    density_fit: bool = False, aux_basis: str = "",
) -> Dict[str, Any]:
    """Decompose a converged ``UMP2Result`` into named energy pieces.

    Open-shell counterpart of :func:`decompose_energy_rmp2`. The
    channel split exposes the aa / bb / ab blocks (UMP2's three
    channels -- same-spin a, same-spin b, opposite-spin) under
    ``channels = {"e_aa": ..., "e_bb": ..., "e_ab": ...}``.

    Returns the same dict shape as :func:`decompose_energy_rmp2`
    with ``method = "ump2"`` and the three-channel split.
    """
    hf_decomp = decompose_energy_uhf(mol, basis, uhf_result, density_fit=False)
    return _mp2_payload(
        "ump2", hf_decomp, ump2_result,
        density_fit=density_fit, aux_basis=aux_basis,
        channel_labels=("e_aa", "e_bb", "e_ab"))
