"""Periodic Γ-point density-fitted 3-index integrals for correlation -- Stage 4.

Exposes the periodic Gaussian-density-fitting (GDF) 3-index tensor ``Lpq`` -- the
fitted "cderi", shape ``(n_fit, nbf, nbf)``, with
``(muν|ls) ≈ S_L Lpq[L,muν] Lpq[L,ls]`` -- so it can be transformed into the
occupied-Wannier / PAO basis for periodic correlation
([`handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md`], Stage 4).

The periodic GDF SCF drivers (`run_pbc_gdf_rhf`, `run_krhf_periodic_gdf`) build
this tensor internally and discard it. This module rebuilds it via the **public**
compcell builder (`vibeqc.aux_basis.build_lpq_compcell`) -- no SCF-driver
internals -- and wraps it in a molecular-`DensityFitting`-compatible interface
(`.three_center`, `.mo_transform`, `.build_J`, `.build_K`).

Correctness is anchored on the SCF: J built from the extracted ``Lpq`` and the
converged density reproduces ``PBCGDFResult.e_coulomb`` (the SCF built its own J
from the same tensor). See `tests/test_periodic_df.py`.

Scope / boundary
----------------
* **Γ-point, compcell GDF** (the default periodic-GDF method). Multi-k ``Lpq(k)``
  exposure and the periodic-MP2 **finite-size / exxdiv-for-correlation** treatment
  (the Nejad 2025 chargeless + surface-dipole corrections -- the G=0 handling that
  makes the correlation energy size-consistent) are Stage 5-6 and are *not* in
  this module. This module only exposes the integral tensor + transforms; it does
  not assert that a naive MP2 over these integrals is the correct periodic
  correlation energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PeriodicGammaDF", "build_periodic_gamma_df"]

# Defaults mirror ``vibeqc.pbc_gdf.run_pbc_gdf_rhf`` so the extracted tensor
# matches the one the GDF SCF builds (the Coulomb-energy verification depends on
# this). If those driver defaults change, the e_coulomb reconstruction test
# fails loudly rather than silently drifting.
_COMPCELL_ETA = 1.0
_APPLY_AFT = True
_AFT_PRECISION = 1e-10
_AFT_FT_CONVENTION = "libint"
_RCUT_STRATEGY = "pyscf_auto"
_RCUT_PRECISION = 1e-8
_GDF_LINDEP_THR = 1e-9


@dataclass
class PeriodicGammaDF:
    """Periodic Γ-point density-fitted 3-index integrals.

    Attributes
    ----------
    three_center : ndarray, shape (n_fit, nbf, nbf)
        The fitted cderi ``Lpq``; ``(muν|ls) ≈ S_L Lpq[L,muν] Lpq[L,ls]``. Each
        ``Lpq[L]`` is symmetric in its AO indices. Real for compcell GDF.
    aux_basis_name : str
        Auxiliary (fitting) basis used.
    n_fit : int
        Number of fit vectors (the L dimension).
    nbf : int
        Number of orbital basis functions.
    """

    three_center: np.ndarray
    aux_basis_name: str
    n_fit: int
    nbf: int

    def build_J(self, D: np.ndarray) -> np.ndarray:
        """Coulomb matrix ``J_muν = S_ls (muν|ls) D_ls`` from the cderi.

        Same contraction as ``pbc_gdf._build_j_from_lpq`` (reproduced here to
        avoid importing a private driver symbol)."""
        L = self.three_center
        cplx = np.iscomplexobj(L)
        Lc = L.conj() if cplx else L
        rho = np.einsum("Lij,ij->L", Lc, D, optimize=True)
        J = np.einsum("L,Lij->ij", rho, L, optimize=True)
        if cplx:
            J = np.real(J)
        return 0.5 * (J + J.T)

    def build_K(self, D: np.ndarray) -> np.ndarray:
        """Exchange matrix ``K_muν = S_κl (muκ|νl) D_κl`` from the cderi.

        This is the bare DF exchange; the periodic ``exxdiv`` (G=0) shift the SCF
        applies on top is *not* included here."""
        L = self.three_center
        cplx = np.iscomplexobj(L)
        Lc = L.conj() if cplx else L
        K = np.einsum("Lmk,kl,Lnl->mn", L, D, Lc, optimize=True)
        if cplx:
            K = np.real(K)
        return 0.5 * (K + K.T)

    def mo_transform(
        self, C_left: np.ndarray, C_right: np.ndarray | None = None
    ) -> np.ndarray:
        """MO-transformed cderi ``B[L,p,q] = S_muν C_left[mu,p] Lpq[L,muν] C_right[ν,q]``.

        ``mo_transform(C_occ, C_vir)`` -> shape ``(n_fit, n_occ, n_vir)``; the MO
        integrals are then ``(pq|rs) = S_L B[L,p,q] B[L,r,s]`` -- the entry point
        for a PNO-basis correlation treatment (Stage 5-6)."""
        if C_right is None:
            C_right = C_left
        tmp = np.einsum("Lmn,nq->Lmq", self.three_center, C_right, optimize=True)
        return np.einsum("mp,Lmq->Lpq", C_left, tmp, optimize=True)


def build_periodic_gamma_df(
    system,
    basis,
    *,
    aux_basis: str | None = None,
    lat_opts=None,
    compcell_eta: float = _COMPCELL_ETA,
    apply_aft_correction: bool = _APPLY_AFT,
    aft_precision: float = _AFT_PRECISION,
    aft_ft_convention: str = _AFT_FT_CONVENTION,
    rcut_strategy=_RCUT_STRATEGY,
    rcut_precision: float = _RCUT_PRECISION,
    gdf_linear_dep_threshold: float = _GDF_LINDEP_THR,
) -> PeriodicGammaDF:
    """Build the periodic Γ-point density-fitted 3-index integrals.

    Uses the public compcell builder ``vibeqc.aux_basis.build_lpq_compcell`` with
    the same defaults as ``vibeqc.pbc_gdf.run_pbc_gdf_rhf``. Pass the *same*
    ``aux_basis`` / ``lat_opts`` / ``compcell_eta`` to this function and to the
    GDF SCF driver to get a matching tensor.

    Parameters
    ----------
    system, basis
        Periodic system + orbital basis (the converged SCF's).
    aux_basis
        Fitting basis name. Defaults to ``default_aux_for(basis.name)`` (the JK
        fit the periodic GDF uses).
    lat_opts
        ``LatticeSumOptions``. If ``None``, ``build_lpq_compcell`` uses its own
        default; pass the SCF's ``options.lattice_opts`` to match its tensor.
    compcell_eta, apply_aft_correction, ... :
        GDF compcell parameters; defaults mirror ``run_pbc_gdf_rhf``.

    Returns
    -------
    PeriodicGammaDF
    """
    from .aux_basis import build_lpq_compcell, default_aux_for, make_aux_basis_set

    mol = system.unit_cell_molecule()
    aux_name = aux_basis or default_aux_for(basis.name)
    aux = make_aux_basis_set(mol, aux_name=aux_name)

    Lpq = np.asarray(
        build_lpq_compcell(
            system,
            basis,
            aux,
            molecule=mol,
            lat_opts=lat_opts,
            linear_dep_thr=float(gdf_linear_dep_threshold),
            compcell_eta=float(compcell_eta),
            apply_aft_correction=bool(apply_aft_correction),
            aft_precision=float(aft_precision),
            aft_ft_convention=str(aft_ft_convention),
            rcut_strategy=rcut_strategy,
            rcut_precision=float(rcut_precision),
        )
    )
    if Lpq.ndim != 3 or Lpq.shape[1] != Lpq.shape[2]:
        raise ValueError(
            f"build_lpq_compcell returned unexpected shape {Lpq.shape}; "
            "expected (n_fit, nbf, nbf)."
        )
    return PeriodicGammaDF(
        three_center=Lpq,
        aux_basis_name=aux_name,
        n_fit=int(Lpq.shape[0]),
        nbf=int(Lpq.shape[1]),
    )
