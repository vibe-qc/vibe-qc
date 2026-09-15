"""Complete active space configuration interaction (CASCI).

CASCI performs a full CI calculation within a user-specified active space
of n electrons in m orbitals, CAS(n,m).  Orbitals outside the active space
are either frozen (doubly occupied, the "core") or empty (the "virtual"
space).  The core contribution enters through a constant energy shift plus
an effective one-electron dressing of the active-space Hamiltonian; the
active-space CI itself is built with the validated unrestricted
Slater-Condon engine (:func:`build_hamiltonian_matrix_unrestricted`), so
CASCI(full space) is identical to FCI.

References
----------
B. O. Roos, P. R. Taylor, P. E. M. Siegbahn, Chem. Phys. 48, 157 (1980).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import eigh

from ._determinant import generate_determinants
from ._slater_condon import (
    build_hamiltonian_matrix_unrestricted,
    diagonal_matrix_element,
)

# Backend dispatch: determinant counts above this go to the C++ direct
# engine (string-based sigma + Davidson, cpp/src/casci.cpp) instead of the
# dense Slater-Condon matrix + eigh.  Force with VIBEQC_CASCI_BACKEND=
# python|cpp|auto.  The direct path lifts the dense ``max_det`` guard up to
# _DIRECT_MAX_DET (the Python determinant-list metadata is the remaining
# memory bound there).
_DIRECT_DET_THRESHOLD = 4000
_DIRECT_MAX_DET = 2_000_000


@dataclass(kw_only=True)
class CASCIOptions:
    """CASCI options for ``run_job(method="casci")``.

    The active space itself comes from ``run_job``'s
    ``active_space=(n_orbitals, n_electrons)`` kwarg (the same convention
    as the CASSCF / CASPT2 siblings); these options carry the remaining
    CASCI knobs.  The standalone solver, :func:`casci`, takes the same
    knobs as explicit keyword arguments instead of this struct.

    Attributes
    ----------
    nroots : int
        Number of CI roots to solve for (default 1).  With ``nroots > 1``
        the headline ``SolverResult.energy`` stays the ground root; the
        per-root total energies land in ``SolverResult.root_energies``
        (per-root <S^2> in ``root_s2``) and the .out solver block prints
        the root table.  Roots are the lowest eigenvectors of the
        fixed-M_s determinant sector with **no** S^2 filtering -- higher-spin
        states interleave (a triplet typically sits between the two lowest
        singlets); the <S^2> column makes the composition visible.
    max_det : int | None
        If set and the active-space determinant count exceeds this, raise
        ValueError rather than building an intractable CI problem
        (default 10 000; determinant spaces past ``_DIRECT_DET_THRESHOLD``
        dispatch to the C++ direct engine, which lifts the practical
        ceiling to ~10⁶ -- raise this deliberately when you mean it).
    """

    nroots: int = 1
    max_det: Optional[int] = 10_000


@dataclass
class CASCIResult:
    """CASCI result.

    Attributes
    ----------
    e_total : float
        Total energy including nuclear repulsion and frozen-core dressing
        (ground state = root 0 when multiple roots are requested).
    e_corr : float
        Correlation energy (E_total - E_HF) for the ground state.
    ci_coeffs : np.ndarray
        CI coefficients in the determinant basis for the ground state
        (shape ``(n_det,)`` when ``nroots=1``; root 0 when ``nroots>1``).
    determinants : list[tuple[tuple[int,...],tuple[int,...]]]
        List of (alpha_occ, beta_occ) determinants in the active space.
    n_active_orb : int
        Number of active spatial orbitals (needed to build RDMs).
    e_core : float
        Constant inactive (frozen-core) energy + nuclear repulsion that was
        added on top of the active-space CI eigenvalue.
    nroots : int
        Number of roots requested (default 1).
    e_totals : list[float]
        Total energies of all requested roots (length ``nroots``).
    ci_coeffs_all : np.ndarray or None
        CI coefficient matrix ``(n_det, nroots)`` when ``nroots > 1``;
        ``None`` when ``nroots == 1`` (use ``ci_coeffs``).
    """

    e_total: float
    e_corr: float
    ci_coeffs: np.ndarray
    determinants: list
    n_det: int = 0
    n_active_orb: int = 0
    e_core: float = 0.0
    nroots: int = 1
    e_totals: list = field(default_factory=list)
    ci_coeffs_all: Optional[np.ndarray] = None


def _frozen_core_dressing(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_core: int,
    active: slice,
) -> tuple[float, np.ndarray]:
    """Inactive (frozen-core) energy and one-electron dressing of the active space.

    Physicist's notation ``g_{pqrs} = <pq|rs> = (pr|qs)``.  For a set of
    doubly-occupied core orbitals ``c``:

      E_core = 2 S_c h_cc + S_{cd} (2 g_{cdcd} - g_{cddc})
      h̃_pq  = h_pq + S_c (2 g_{pcqc} - g_{pccq})    (p, q active)

    ``E_core`` is exactly the closed-shell diagonal element of the core
    determinant (:func:`diagonal_matrix_element`), and the Coulomb /
    exchange dressing is the mean field of the doubly-occupied core.
    """
    if n_core == 0:
        return 0.0, h1e_mo[active, active].copy()

    e_core = diagonal_matrix_element(tuple(range(n_core)), h1e_mo, h2e_mo)

    # g_{pcqc} (Coulomb) and g_{pccq} (exchange), summed over core c.
    coulomb = np.einsum("pcqc->pq", h2e_mo[active, :n_core, active, :n_core])
    exchange = np.einsum("pccq->pq", h2e_mo[active, :n_core, :n_core, active])
    h1e_active = h1e_mo[active, active] + 2.0 * coulomb - exchange
    return e_core, h1e_active


def casci(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_active_elec: int,
    n_active_orb: int,
    n_core: int = 0,
    nuclear_repulsion: float = 0.0,
    e_hf: float = 0.0,
    ms2: Optional[int] = None,
    *,
    nroots: int = 1,
    active_orbitals: Optional[list[int]] = None,
    max_det: Optional[int] = 10_000,
    ci_guess: Optional[np.ndarray] = None,
) -> CASCIResult:
    """Run CASCI in an active space.

    Parameters
    ----------
    h1e_mo : (norb, norb) ndarray
        One-electron Hamiltonian in the MO basis.
    h2e_mo : (norb, norb, norb, norb) ndarray
        Two-electron integrals in physicist's notation, MO basis.
    n_active_elec : int
        Number of active electrons.
    n_active_orb : int
        Number of active spatial orbitals.
    n_core : int
        Number of doubly-occupied inactive (core) orbitals.
    nuclear_repulsion : float
        Nuclear repulsion energy.
    e_hf : float
        Reference HF energy, used only to report ``e_corr``.
    ms2 : int, optional
        ``2 S_z = n_alpha \u2212 n_beta`` for the active electrons.  Defaults to
        ``n_active_elec % 2`` (closed-shell singlet / minimal-spin doublet).
    nroots : int
        Number of CI roots to compute (default 1).  The ground state is root 0.
    active_orbitals : list[int], optional
        Explicit 0-based orbital indices for the active space.  Must be
        contiguous.  When provided, ``n_active_orb`` and ``n_core`` are
        derived from it.  Enables CAS-window selection (e.g., d-shell).
    ci_guess : ndarray, optional
        ``(n_det,)`` or ``(n_det, k)`` warm-start vector(s) for the direct
        (Davidson) backend -- e.g. the previous macro-iteration's CI
        vector(s) in a CASSCF loop.  Ignored by the dense backend (exact
        ``eigh``).

    Returns
    -------
    CASCIResult
    """
    norb_total = h1e_mo.shape[0]

    # Resolve active-orbital indices.
    if active_orbitals is not None:
        act_idx = sorted(active_orbitals)
        if not act_idx:
            raise ValueError("active_orbitals must not be empty")
        expected = list(range(act_idx[0], act_idx[-1] + 1))
        if act_idx != expected:
            raise ValueError(
                f"active_orbitals must be contiguous; got gap in {act_idx}"
            )
        n_core_actual = act_idx[0]
        n_active_orb_actual = len(act_idx)
        active_slice = slice(act_idx[0], act_idx[-1] + 1)
    else:
        n_core_actual = n_core
        n_active_orb_actual = n_active_orb
        active_slice = slice(n_core, n_core + n_active_orb)

    n_virt = norb_total - n_core_actual - n_active_orb_actual
    if n_virt < 0:
        raise ValueError(
            f"n_core ({n_core_actual}) + n_active_orb ({n_active_orb_actual}) > norb ({norb_total})"
        )

    if ms2 is None:
        ms2 = n_active_elec % 2
    n_alpha = (n_active_elec + ms2) // 2
    n_beta = n_active_elec - n_alpha
    if (
        n_alpha < 0
        or n_beta < 0
        or n_alpha > n_active_orb_actual
        or n_beta > n_active_orb_actual
    ):
        raise ValueError(
            f"Cannot place {n_active_elec} electrons (ms2={ms2}) in "
            f"{n_active_orb_actual} active orbitals"
        )
    if nroots < 1:
        raise ValueError(f"nroots must be >= 1, got {nroots}")

    e_core, h1e_active = _frozen_core_dressing(
        h1e_mo, h2e_mo, n_core_actual, active_slice
    )
    h2e_active = np.ascontiguousarray(
        h2e_mo[active_slice, active_slice, active_slice, active_slice]
    )

    from math import comb

    n_det = comb(n_active_orb_actual, n_alpha) * comb(n_active_orb_actual, n_beta)
    if n_det == 0:
        raise ValueError(
            "No determinants in the active space \u2014 check n_active_elec / n_active_orb"
        )
    if nroots > n_det:
        raise ValueError(f"nroots ({nroots}) exceeds number of determinants ({n_det})")

    backend = os.environ.get("VIBEQC_CASCI_BACKEND", "auto")
    if backend not in ("auto", "cpp", "python"):
        raise ValueError(
            f"VIBEQC_CASCI_BACKEND must be auto|cpp|python, got {backend!r}"
        )
    use_direct = backend == "cpp" or (
        backend == "auto" and n_det > _DIRECT_DET_THRESHOLD
    )
    if use_direct:
        try:
            from .._vibeqc_core import CASCIDirectOptions as _DirectOpts
            from .._vibeqc_core import casci_direct_solve as _direct_solve
        except ImportError:  # extension predates the direct engine
            use_direct = False
            if backend == "cpp":
                raise

    if use_direct:
        if n_det > _DIRECT_MAX_DET:
            raise ValueError(
                f"CASCI determinant space ({n_det}) exceeds the direct-engine "
                f"limit ({_DIRECT_MAX_DET})."
            )
    elif max_det is not None and n_det > max_det:
        raise ValueError(
            f"CASCI determinant space ({n_det}) exceeds max_det ({max_det}). "
            "Reduce the active space or increase max_det."
        )

    determinants = generate_determinants(n_active_orb_actual, n_alpha, n_beta)
    assert len(determinants) == n_det

    if use_direct:
        # Direct string-based engine: chemist (pq|rs) = g_phys[p,r,q,s];
        # determinant ordering and phases match the dense engine by
        # construction (see cpp/include/vibeqc/casci.hpp).
        opts = _DirectOpts()
        opts.nroots = nroots
        eri_chem = np.ascontiguousarray(h2e_active.transpose(0, 2, 1, 3))
        guess = np.empty((0, 0))
        if ci_guess is not None:
            guess = np.asarray(ci_guess, dtype=float)
            if guess.ndim == 1:
                guess = guess[:, None]
            if guess.shape[0] != n_det:
                raise ValueError(
                    f"ci_guess has {guess.shape[0]} rows; the determinant "
                    f"space has {n_det}"
                )
            guess = np.ascontiguousarray(guess)
        direct = _direct_solve(
            np.ascontiguousarray(h1e_active),
            eri_chem,
            n_active_orb_actual,
            n_alpha,
            n_beta,
            opts,
            guess,
        )
        if not direct.converged:
            raise RuntimeError(
                f"Direct CAS-CI Davidson did not converge in {direct.n_iter} "
                "iterations; force VIBEQC_CASCI_BACKEND=python to fall back "
                "to the dense engine (if the space is small enough)."
            )
        eigvals = np.asarray(direct.eigenvalues)
        eigvecs = np.asarray(direct.ci)
    else:
        ham = build_hamiltonian_matrix_unrestricted(
            determinants, h1e_active, h2e_active
        )
        eigvals, eigvecs = eigh(ham)

    e_const = e_core + nuclear_repulsion
    e_total = float(eigvals[0]) + e_const

    e_totals = [float(eigvals[i]) + e_const for i in range(nroots)]
    ci_all = eigvecs[:, :nroots] if nroots > 1 else None

    return CASCIResult(
        e_total=e_total,
        e_corr=e_total - e_hf,
        ci_coeffs=eigvecs[:, 0].copy(),
        determinants=determinants,
        n_det=n_det,
        n_active_orb=n_active_orb_actual,
        e_core=e_const,
        nroots=nroots,
        e_totals=e_totals,
        ci_coeffs_all=ci_all,
    )
