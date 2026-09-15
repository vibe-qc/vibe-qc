"""Time-dependent DFT -- Casida linear-response and Tamm-Dancoff approximation.

Computes vertical excitation energies, oscillator strengths, and
transition dipole moments from a converged ground-state SCF result
using the linear-response TDDFT formalism of Casida.

**Theory (Casida, 1995).** The excitation energies w are eigenvalues
of the symplectic eigenvalue problem:

    ``(A - B)^{1/2} (A + B) (A - B)^{1/2} Z = w^2 Z``

where the orbital-rotation Hessian matrices in the canonical MO basis are:

    A_{ia,jb} = d_{ij} d_{ab} (e_a - e_i) + 2 (ia|jb) - c_x (ij|ab) + (ia|f_xc|jb)
    B_{ia,jb} = 2 (ia|jb) - c_x (ib|ja) + (ia|f_xc|jb)

For TDA (Tamm-Dancoff approximation, equivalent to CIS when f_xc = 0),
the B matrix is neglected and the eigenvalue problem reduces to:

    A X = w X

For RHF (c_x = 1) the exchange term is included in full; for pure DFT
(c_x = 0) it drops out and only the f_xc kernel remains (ALDA). For a
global hybrid, c_x is the functional's exact-exchange admixture a
(B3LYP 0.20, PBE0 0.25, ...) and f_xc is the adiabatic kernel of the
functional's DFT part -- the standard hybrid TDDFT kernel
(Bauernschmitt & Ahlrichs, 1996). Range-separated hybrids are refused:
their position-dependent exchange a + b.erf(w.r) is not representable
by a single global c_x over regular ERIs.

**References**

* Casida, M. E. "Time-Dependent Density Functional Response Theory for
  Molecules." In *Recent Advances in Density Functional Methods*, Part I,
  ed. D. P. Chong, 155-192 (World Scientific, 1995).
* Bauernschmitt, R. & Ahlrichs, R. *Chem. Phys. Lett.* **256**, 454
  (1996). (Adiabatic-approximation TDDFT excitations incl. the hybrid
  kernel composition a.HF-exchange + f_xc.)
* Dreuw, A. & Head-Gordon, M. *Chem. Rev.* **105**, 4009 (2005).
  (Single-reference ab initio methods for excited states -- review.)
* Hirata, S. & Head-Gordon, M. *Chem. Phys. Lett.* **314**, 291 (1999).
  (TDA benchmark; TDA-B3LYP reference values used for validation.)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    Functional,
    GridOptions,
    Molecule,
    build_grid,
    compute_dipole,
    compute_eri,
    eri_ao_to_mo_blocks,
    evaluate_ao,
    make_polarised_xc_kernel_builder,
    make_unpolarised_xc_kernel_builder,
)

__all__ = [
    "NTOResult",
    "TDDFTResult",
    "TDDFTState",
    "compute_nto",
    "compute_nto_uhf",
    "eri_ao_to_mo",
    "eri_ao_to_mo_blocks",
    "make_eri_provider",
    "oscillator_strength",
    "run_tddft_casida",
    "run_tddft_casida_uhf",
    "run_tddft_tda",
    "run_tddft_tda_periodic",
    "run_tddft_tda_rohf",
    "run_tddft_tda_uhf",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TDDFTState:
    """A single TDDFT excited state.

    Attributes
    ----------
    index : int
        State number (1-based).
    excitation_energy : float
        Vertical excitation energy in Hartree.
    excitation_energy_ev : float
        Vertical excitation energy in eV.
    wavelength_nm : float
        Excitation wavelength in nanometres.
    oscillator_strength : float
        Dimensionless oscillator strength (length gauge).
    transition_dipole : np.ndarray
        Transition dipole moment (x, y, z) in a.u. (e.bohr).
    dominant_amplitudes : list of tuple
        Leading (occ->virt, |amplitude|) pairs, sorted by |amplitude|.
    excitation_vector : np.ndarray
        The full (n_occ x n_virt,) eigenvector X (TDA) or X+Y (Casida).
    """

    index: int
    excitation_energy: float
    excitation_energy_ev: float
    wavelength_nm: float
    oscillator_strength: float
    transition_dipole: np.ndarray
    dominant_amplitudes: List[Tuple[int, int, float]] = field(default_factory=list)
    excitation_vector: Optional[np.ndarray] = None


@dataclass
class TDDFTResult:
    """Container for a full TDDFT calculation.

    Attributes
    ----------
    n_states : int
        Number of excited states requested.
    n_occ : int
        Number of occupied orbitals.
    n_virt : int
        Number of virtual orbitals.
    method : str
        ``"TDA"`` or ``"Casida"``.
    functional : str or None
        XC functional name (``None`` for RHF/UHF).
    states : list of TDDFTState
        Computed excited states, sorted by energy.
    """

    n_states: int
    n_occ: int
    n_virt: int
    method: str
    functional: Optional[str]
    states: List[TDDFTState] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AO -> MO ERI transformation
# ---------------------------------------------------------------------------


def eri_ao_to_mo(
    eri_ao: np.ndarray,
    mo_coeff: np.ndarray,
    n_occ: int,
) -> np.ndarray:
    """Transform 4-index AO ERIs to the MO basis, returning the
    (ia|jb) block needed for TDDFT.

    The full MO-basis 4-index integrals are:

        (pq|rs) = S_{muνls} C_{mup} C_{νq} C_{lr} C_{ss} (muν|ls)

    We only need the occ-virt block: orbitals i, j in [0, n_occ) and
    a, b in [n_occ, n_basis).

    Parameters
    ----------
    eri_ao : np.ndarray
        AO-basis 4-index ERI tensor of shape ``(n_basis, n_basis,
        n_basis, n_basis)``.
    mo_coeff : np.ndarray
        MO coefficient matrix of shape ``(n_basis, n_basis)``
        (columns = MOs).
    n_occ : int
        Number of occupied orbitals.

    Returns
    -------
    eri_mo_ovov : np.ndarray
        ``(n_occ, n_virt, n_occ, n_virt)`` array of ``(ia|jb)``
        integrals.
    """
    n_basis = mo_coeff.shape[0]
    n_virt = n_basis - n_occ

    # Occ and virt coefficient blocks
    C_occ = np.ascontiguousarray(mo_coeff[:, :n_occ])  # (n_basis, n_occ)
    C_virt = np.ascontiguousarray(mo_coeff[:, n_occ:])  # (n_basis, n_virt)

    # Step 1: half-transform index 0 (mu) -> occ (i)
    # C_occ axes: (basis=mu, occ=i); eri_ao axes: (mu, nu, lam, sig)
    eri_h1 = np.einsum("mi,mnls->inls", C_occ, eri_ao, optimize=False)
    # eri_h1 shape: (n_occ, n_basis, n_basis, n_basis)

    # Step 2: half-transform index 1 (nu) -> virt (a)
    # C_virt axes: (basis=nu, virt=a); eri_h1 axes: (i, nu, lam, sig)
    eri_h2 = np.einsum("na,inls->ials", C_virt, eri_h1, optimize=False)
    # eri_h2 shape: (n_occ, n_virt, n_basis, n_basis)

    # Step 3: half-transform index 2 (lam) -> occ (j)
    # C_occ axes: (basis=lam, occ=j); eri_h2 axes: (i, a, lam, sig)
    eri_h3 = np.einsum("lj,ials->iajs", C_occ, eri_h2, optimize=False)
    # eri_h3 shape: (n_occ, n_virt, n_occ, n_basis)

    # Step 4: half-transform index 3 (sig) -> virt (b)
    # C_virt axes: (basis=sig, virt=b); eri_h3 axes: (i, a, j, sig)
    eri_mo = np.einsum("sb,iajs->iajb", C_virt, eri_h3, optimize=False)
    # eri_mo shape: (n_occ, n_virt, n_occ, n_virt) = (i,a|j,b)
    return eri_mo


def eri_ao_to_mo_cross(
    eri_ao: np.ndarray,
    mo_coeff_alpha: np.ndarray,
    mo_coeff_beta: np.ndarray,
    n_occ_alpha: int,
    n_occ_beta: int,
) -> np.ndarray:
    """Cross-spin ``(i_alpha a_alpha | j_beta b_beta)`` ERI block."""
    n_virt_alpha = mo_coeff_alpha.shape[1] - n_occ_alpha
    n_virt_beta = mo_coeff_beta.shape[1] - n_occ_beta
    if (
        n_occ_alpha <= 0
        or n_virt_alpha <= 0
        or n_occ_beta <= 0
        or n_virt_beta <= 0
    ):
        raise ValueError(
            "eri_ao_to_mo_cross: occupied and virtual spaces must both be "
            "non-empty for alpha and beta spins"
        )

    C_occ_alpha = np.ascontiguousarray(mo_coeff_alpha[:, :n_occ_alpha])
    C_virt_alpha = np.ascontiguousarray(mo_coeff_alpha[:, n_occ_alpha:])
    C_occ_beta = np.ascontiguousarray(mo_coeff_beta[:, :n_occ_beta])
    C_virt_beta = np.ascontiguousarray(mo_coeff_beta[:, n_occ_beta:])

    h1 = np.einsum("mi,mnls->inls", C_occ_alpha, eri_ao, optimize=False)
    h2 = np.einsum("na,inls->ials", C_virt_alpha, h1, optimize=False)
    h3 = np.einsum("lj,ials->iajs", C_occ_beta, h2, optimize=False)
    return np.einsum("sb,iajs->iajb", C_virt_beta, h3, optimize=False)


def _eri_oovv(eri_ao, mo_coeff, n_occ):
    """Compute the (ij|ab) MO ERI block for the exchange kernel."""
    C_occ = np.ascontiguousarray(mo_coeff[:, :n_occ])
    C_virt = np.ascontiguousarray(mo_coeff[:, n_occ:])
    h1 = np.einsum("mi,mnls->inls", C_occ, eri_ao, optimize=False)
    h2 = np.einsum("nj,inls->ijls", C_occ, h1, optimize=False)
    h3 = np.einsum("ra,ijrs->ijas", C_virt, h2, optimize=False)
    h4 = np.einsum("sb,ijas->ijab", C_virt, h3, optimize=False)
    return h4


def make_eri_provider(
    eri_ao: np.ndarray,
    mo_coeff: np.ndarray,
    n_occ: int,
):
    """Build an ``ERIProvider`` from AO integrals + MO coefficients.

    Returns an object compatible with :class:`vibeqc.correlation.ERIProvider`
    that provides ``ovov()`` and ``oovv()`` MO-basis integral blocks.
    Useful for feeding TDDFT integrals into MP2, OVGF, and other
    correlated-method backends.
    """
    from dataclasses import dataclass

    @dataclass
    class _LazyERIProvider:
        _eri_ao: np.ndarray
        _mo_coeff: np.ndarray
        _n_occ: int
        _ovov_cache: np.ndarray | None = None
        _oovv_cache: np.ndarray | None = None

        def ovov(self) -> np.ndarray:
            if self._ovov_cache is None:
                self._ovov_cache = eri_ao_to_mo(
                    self._eri_ao,
                    self._mo_coeff,
                    self._n_occ,
                )
            return self._ovov_cache

        def oovv(self) -> np.ndarray:
            if self._oovv_cache is None:
                self._oovv_cache = _eri_oovv(
                    self._eri_ao,
                    self._mo_coeff,
                    self._n_occ,
                )
            return self._oovv_cache

    return _LazyERIProvider(
        _eri_ao=np.asarray(eri_ao),
        _mo_coeff=np.asarray(mo_coeff),
        _n_occ=n_occ,
    )


# ---------------------------------------------------------------------------
def _resolve_exact_exchange_fraction(functional: Optional[str]) -> float:
    """Exact-exchange coefficient c_x of the response kernel.

    ``None`` (HF reference, CIS/TDHF) -> 1.0; pure functionals -> 0.0;
    global hybrids -> their exact-exchange admixture a (B3LYP 0.20,
    PBE0 0.25, ...) from ``Functional.hf_exchange_fraction``, composing
    the standard hybrid TDDFT kernel a.HF-exchange + f_xc[DFT part]
    of Bauernschmitt & Ahlrichs, Chem. Phys. Lett. 256, 454 (1996),
    doi:10.1016/0009-2614(96)00440-X.

    Range-separated hybrids are refused rather than run wrong: their
    position-dependent exchange a + b.erf(w.r) requires erf-attenuated
    (ij|ab) integrals, not a single global scale on regular ERIs.
    """
    if functional is None:
        return 1.0
    func = Functional(functional, 1)
    if func.is_range_separated:
        raise NotImplementedError(
            f"TDDFT/TDA with the range-separated hybrid '{functional}' "
            f"is not implemented: its exact exchange a + b.erf(w.r) "
            f"(a = {func.cam_alpha:g}, b = {func.cam_beta:g}, w = "
            f"{func.rsh_omega:g} bohr⁻¹) cannot be represented by a "
            "single global exchange coefficient over regular ERIs. "
            "Use a global hybrid (pbe0, b3lyp, ...) or a pure functional."
        )
    if func.is_double_hybrid:
        warnings.warn(
            f"TDDFT/TDA with the double hybrid '{functional}': the "
            "perturbative (D) doubles correction is not implemented; "
            "excitation energies are the bare TD response of the "
            "hybrid reference (exact exchange "
            f"c_x = {func.hf_exchange_fraction:g} + f_xc).",
            UserWarning,
            stacklevel=3,
        )
    return float(func.hf_exchange_fraction)


def _warn_hybrid_without_fxc(functional: str, c_x: float, detail: str) -> None:
    """Flag the partial-kernel mode: a.HF exchange without f_xc is not
    the full TDA/TDDFT-<functional> kernel and must not pass silently."""
    warnings.warn(
        f"TDDFT/TDA with hybrid functional '{functional}': the exact-"
        f"exchange term (c_x = {c_x:g}) is included, but the f_xc "
        f"kernel is not ({detail}). Excitation energies are NOT full "
        f"TDA/TDDFT-{functional} values.",
        UserWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
def _compute_alda_kernel_mo(
    molecule: Molecule,
    basis: BasisSet,
    functional_name: str,
    mo_coeff: np.ndarray,
    density_ao: np.ndarray,
    n_occ: int,
    *,
    grid_options: Optional[GridOptions] = None,
) -> np.ndarray:
    """Compute the ALDA XC kernel matrix in the MO (occ,virt,occ,virt) basis.

    For each occ-virt pair (j,b), builds the density perturbation
    D^{jb} = c_j c_b^T, applies the XC kernel builder to get
    W^{XC}[D^{jb}] in the AO basis, then projects onto the MO basis.

    Returns (n_occ, n_virt, n_occ, n_virt) array of (ia|f_xc|jb).

    This function only supports LDA and GGA functionals (the ALDA /
    AGGA kernel). meta-GGA kernels are not yet available.
    """
    n_basis = mo_coeff.shape[0]
    n_virt = n_basis - n_occ

    # Build grid and evaluate AOs
    if grid_options is None:
        grid_options = GridOptions()
    grid = build_grid(molecule, grid_options)
    chi_tuple = evaluate_ao(basis, grid.points)
    # evaluate_ao_with_gradient returns (chi, dchi_x, dchi_y, dchi_z)
    # but evaluate_ao just returns chi. The kernel builder needs
    # evaluate_ao_with_gradient for GGA.
    # For ALDA, dchi can be zero-filled.
    from ._vibeqc_core import evaluate_ao_with_gradient

    chi, dchi_x, dchi_y, dchi_z = evaluate_ao_with_gradient(basis, grid.points)

    # Build functional and XC kernel builder
    func = Functional(functional_name, 1)  # 1 = unpolarised
    builder = make_unpolarised_xc_kernel_builder(
        func,
        grid,
        chi,
        dchi_x,
        dchi_y,
        dchi_z,
        np.asarray(density_ao, dtype=np.float64),
    )

    # Compute kernel matrix: (ia|f_xc|jb)
    kernel_mo = np.zeros((n_occ, n_virt, n_occ, n_virt))
    C_occ = np.ascontiguousarray(mo_coeff[:, :n_occ])
    C_virt = np.ascontiguousarray(mo_coeff[:, n_occ:])

    for j in range(n_occ):
        c_j = C_occ[:, j]
        for b in range(n_virt):
            c_b = C_virt[:, b]
            # Density perturbation: c_j c_b^T (non-symmetric; builder
            # expects symmetric, so we symmetrise)
            D_jb = np.outer(c_j, c_b) + np.outer(c_b, c_j)
            # Apply XC kernel
            W_xc = np.asarray(builder.apply(D_jb))
            # Project: c_i^T W_xc c_a for all i,a
            W_C_occ = C_occ.T @ W_xc  # (n_occ, n_basis)
            for i in range(n_occ):
                for a in range(n_virt):
                    kernel_mo[i, a, j, b] = float(W_C_occ[i, :] @ C_virt[:, a])

    return kernel_mo


# ---------------------------------------------------------------------------
# Oscillator strength
# ---------------------------------------------------------------------------


def oscillator_strength(
    excitation_energy: float,
    transition_dipole: np.ndarray,
) -> float:
    """Compute the dimensionless oscillator strength in the length gauge.

    Parameters
    ----------
    excitation_energy : float
        Excitation energy in Hartree.
    transition_dipole : np.ndarray
        Transition dipole moment vector ``(3,)`` in a.u. (e.bohr).

    Returns
    -------
    f : float
        Oscillator strength (dimensionless).

    Notes
    -----
    f = (2/3) . w . |mu|^2  where w is the excitation energy and mu
    is the transition dipole moment, both in atomic units. This is
    the dipole-length form; good to ~5% for valence excitations.
    """
    mu_sq = float(np.dot(transition_dipole, transition_dipole))
    return (2.0 / 3.0) * excitation_energy * mu_sq


# ---------------------------------------------------------------------------
# Transition density -> transition dipole
# ---------------------------------------------------------------------------


def _transition_dipole_from_amplitudes(
    X: np.ndarray,
    n_occ: int,
    n_virt: int,
    mo_coeff: np.ndarray,
    dipole_ao,  # DipoleIntegrals with .x, .y, .z attributes
) -> np.ndarray:
    """Compute the transition dipole moment from the excitation vector.

    For a single-determinant ground state, the transition density between
    the ground state |0> and excited state |I> in the MO basis is:

        T_{pq} = S_{ia} X_{ia}^{I} (d_{pi} d_{qa} + d_{pa} d_{qi})

    where the off-diagonal blocks are the excitation amplitudes. In the
    AO basis:

        P_{muν}^{0I} = S_{ia} X_{ia}^{I} (C_{mui} C_{νa} + C_{mua} C_{νi})
                    = C_occ . X . C_virt^T + C_virt . X^T . C_occ^T

    The transition dipole moment is tr(P^{0I} . mu) in each cartesian
    direction.
    """
    C_occ = mo_coeff[:, :n_occ]  # (n_basis, n_occ)
    C_virt = mo_coeff[:, n_occ:]  # (n_basis, n_virt)

    # Reshape the (n_occ * n_virt,) vector to (n_occ, n_virt)
    X_ia = X.reshape(n_occ, n_virt)

    # AO transition density
    P_ao = C_occ @ X_ia @ C_virt.T + C_virt @ X_ia.T @ C_occ.T

    mu = np.zeros(3)
    for d, comp in enumerate([dipole_ao.x, dipole_ao.y, dipole_ao.z]):
        mu[d] = np.trace(P_ao @ np.asarray(comp))
    return mu


# ---------------------------------------------------------------------------
# Dominant amplitudes
# ---------------------------------------------------------------------------


def _dominant_amplitudes(
    X: np.ndarray,
    n_occ: int,
    n_virt: int,
    top_n: int = 5,
) -> List[Tuple[int, int, float]]:
    """Extract the leading (occ, virt) contributions from the
    excitation vector, sorted by absolute amplitude.

    Returns a list of ``(occ_index, virt_index, amplitude)`` tuples.
    Indices are 1-based (orbital numbering convention).
    """
    X_ia = X.reshape(n_occ, n_virt)
    abs_ia = np.abs(X_ia)
    # Flatten and sort
    idx_flat = np.argsort(-abs_ia.ravel())
    result = []
    for k in idx_flat[:top_n]:
        i = int(k) // n_virt
        a = int(k) % n_virt
        amp = float(X_ia[i, a])
        result.append((i + 1, a + 1, amp))
    return result


# ---------------------------------------------------------------------------
# TDA solver
# ---------------------------------------------------------------------------


def _build_tda_matrix(
    mo_energies: np.ndarray,
    eri_mo_ovov: np.ndarray,
    eri_mo_oovv: np.ndarray,
    n_occ: int,
    c_x: float,
    kernel_mo: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the TDA (Casida A) matrix.

    A_{ia,jb} = delta_ij delta_ab (eps_a - eps_i) + 2 (ia|jb) - c_x (ij|ab)
    """
    n_virt = mo_energies.shape[0] - n_occ
    n_pair = n_occ * n_virt

    eps_occ = mo_energies[:n_occ]
    eps_virt = mo_energies[n_occ:]

    # (ia|jb) already sits in composite-index order: reshaping
    # (n_occ, n_virt, n_occ, n_virt) -> (n_pair, n_pair) maps (i, a) to the row
    # and (j, b) to the column, which is exactly A's layout.
    A = 2.0 * eri_mo_ovov.reshape(n_pair, n_pair)

    # (ij|ab) is stored (i, j, a, b); the exchange term needs it as
    # (i, a, j, b), hence the transpose before the same reshape.
    if c_x != 0.0:
        A -= c_x * eri_mo_oovv.transpose(0, 2, 1, 3).reshape(n_pair, n_pair)

    if kernel_mo is not None:
        A += kernel_mo.reshape(n_pair, n_pair)

    # delta_ij delta_ab (eps_a - eps_i) -- the orbital-energy differences.
    A[np.diag_indices(n_pair)] += (
        eps_virt[None, :] - eps_occ[:, None]
    ).reshape(n_pair)

    return A


def tda_sigma_and_diagonal(
    mo_energies: np.ndarray,
    eri_mo_ovov: np.ndarray,
    eri_mo_oovv: np.ndarray,
    n_occ: int,
    c_x: float,
    kernel_mo: Optional[np.ndarray] = None,
):
    """Matrix-free application of the TDA (Casida A) operator.

    Returns ``(sigma, diagonal)`` where ``sigma(X)`` computes ``A @ X`` for a
    flat trial vector (or a block of them, column-wise) without ever forming
    ``A``, and ``diagonal`` is A's diagonal in closed form.

    ``_build_tda_matrix`` materialises A explicitly, which costs
    ``(n_occ*n_virt)**2`` doubles *on top of* the ``eri_mo_ovov`` tensor of the
    same size, and a dense diagonalisation is then ``O(dim**3)``.  For
    adenine-thymine/cc-pVDZ that is a 17,204-dimensional matrix -- 2.2 GB and
    ~5e12 flops -- to obtain the ~10 lowest roots.  Iterating instead needs
    only repeated ``A @ X``, which this provides at ``O(dim**2)`` per vector
    with no ``A`` in memory.

    The operator is the same one ``_build_tda_matrix`` writes down:

        A_{ia,jb} = delta_ij delta_ab (eps_a - eps_i)
                    + 2 (ia|jb) - c_x (ij|ab) + (ia|f_xc|jb)

    so, for a trial amplitude ``X_{jb}``,

        (A X)_{ia} = (eps_a - eps_i) X_{ia}
                     + sum_{jb} [2 (ia|jb) - c_x (ij|ab) + (ia|f_xc|jb)] X_{jb}

    and the diagonal follows by setting (jb) = (ia).  The two forms are pinned
    against each other in ``tests/test_tddft_davidson.py``; that equality is
    the correctness contract, since an iterative solve must reproduce the dense
    one it replaces.

    The diagonal is returned rather than probed.  ``EigenProblem`` will
    otherwise recover it by applying the operator to ``n`` unit vectors, which
    is ``O(dim**3)`` and would cost more than the dense diagonalisation this
    exists to avoid.
    """
    n_virt = mo_energies.shape[0] - n_occ
    n_pair = n_occ * n_virt
    eps_occ = mo_energies[:n_occ]
    eps_virt = mo_energies[n_occ:]

    # Orbital-energy difference, shaped (n_occ, n_virt).
    delta_eps = eps_virt[None, :] - eps_occ[:, None]

    has_kernel = kernel_mo is not None

    def sigma(v: np.ndarray) -> np.ndarray:
        """A @ v for a single vector (n_pair,) or a block (n_pair, k)."""
        vec = np.asarray(v, dtype=float)
        flat = vec.ndim == 1
        block = vec.reshape(n_pair, -1)
        out = np.empty_like(block)
        for col in range(block.shape[1]):
            X = block[:, col].reshape(n_occ, n_virt)
            AX = delta_eps * X
            AX += 2.0 * np.einsum("iajb,jb->ia", eri_mo_ovov, X, optimize=True)
            AX -= c_x * np.einsum("ijab,jb->ia", eri_mo_oovv, X, optimize=True)
            if has_kernel:
                AX += np.einsum("iajb,jb->ia", kernel_mo, X, optimize=True)
            out[:, col] = AX.reshape(n_pair)
        return out.reshape(vec.shape) if flat else out

    # Closed-form diagonal: the (ia),(ia) element of the expression above.
    diag = delta_eps.copy()
    diag += 2.0 * np.einsum("iaia->ia", eri_mo_ovov, optimize=True)
    diag -= c_x * np.einsum("iiaa->ia", eri_mo_oovv, optimize=True)
    if has_kernel:
        diag += np.einsum("iaia->ia", kernel_mo, optimize=True)

    return sigma, diag.reshape(n_pair)


#: TDA problems at or below this dimension are diagonalised densely; above it
#: the lowest roots are found iteratively.  Measured, cc-pVDZ, 8 roots, dense
#: (vectorised build + ``eigh``) against matrix-free Davidson:
#:
#:     dim   380   dense 0.01 s   iterative 0.03 s   0.38x  -- dense wins
#:     dim   480   dense 0.03 s   iterative 0.09 s   0.29x  -- dense wins
#:     dim  1920   dense 1.87 s   iterative 1.49 s   1.25x
#:     dim  2912   dense 5.80 s   iterative 3.12 s   1.86x
#:     dim  2987   dense 6.03 s   iterative 4.49 s   1.34x
#:
#: so the crossover sits between 480 and 1920 and 2000 is the conservative
#: side of it.  The margin grows with dimension: dense is ``O(dim**3)`` and
#: allocates ``dim**2`` doubles *on top of* an ``eri_mo_ovov`` tensor of the
#: same size, while the iterative path is ``O(dim**2)`` per operator
#: application at a root-count that does not grow with the system and forms no
#: matrix at all.  For adenine-thymine/cc-pVDZ (dim 17,204) that is a 2.2 GB
#: allocation the iterative path does not make.
#:
#: Both paths solve the same operator and agree to ~1e-12 Ha;
#: tests/test_tddft_davidson.py pins them against each other.
TDA_DAVIDSON_MIN_DIM = 2000

#: Only iterate when the requested roots are a small fraction of the spectrum;
#: asking for most of it is what dense diagonalisation is for.
TDA_DAVIDSON_MAX_ROOT_FRACTION = 0.1


def _tda_guess_vectors(diagonal: np.ndarray, n_guess: int) -> np.ndarray:
    """Unit vectors on the ``n_guess`` lowest diagonal elements.

    The conventional response-theory starting guess: A's diagonal is dominated
    by the orbital-energy differences, so its smallest entries are the
    single excitations that dominate the lowest roots.

    This is also a *correctness* requirement here, not only an efficiency one.
    GitLab #503: with no explicit guess the shared Davidson runs at
    ``n_guess=0`` and, for a small ``n_eig``, can report ``converged=True``
    while returning the wrong roots -- reproducibly 4.0 Ha out on a 200-
    dimensional test operator.  Supplying a guess moves that to 4.3e-13.
    """
    k = max(1, min(int(n_guess), diagonal.shape[0]))
    idx = np.argsort(diagonal)[:k]
    guess = np.zeros((diagonal.shape[0], k), dtype=float)
    guess[idx, np.arange(k)] = 1.0
    return guess


def _solve_tda_iterative(
    sigma,
    diagonal: np.ndarray,
    n_pair: int,
    n_requested: int,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
):
    """Lowest ``n_requested`` roots of the TDA operator, matrix-free.

    Returns ``(eigenvalues, eigenvectors, max_residual)``.

    The residual is recomputed here from ``sigma`` rather than read off the
    solver.  Per GitLab #503 the shared Davidson does not signal convergence
    for a partial spectrum -- it returns ``converged=False`` and runs its full
    iteration budget even when the roots are accurate to 1e-13 -- so its flag
    carries no information either way.  ``n_requested`` extra operator
    applications buy a number we can actually act on.
    """
    from .solvers.eigensolver import (
        EigenProblem,
        SolverOptions,
        solve_eigenproblem,
    )

    problem = EigenProblem(
        matvec=sigma,
        n=n_pair,
        diagonal=diagonal,
    )
    options = SolverOptions(
        n_roots=n_requested,
        which="SA",
        tol=tol,
        max_iter=max_iter,
    )
    # Twice the requested roots is the usual response-theory block size; it
    # also keeps #503's wrong-root mode out of reach.
    options.guess_vectors = _tda_guess_vectors(diagonal, 2 * n_requested)

    result = solve_eigenproblem(problem, options, solver="davidson")

    evals = np.asarray(result.eigenvalues, dtype=float)
    evecs = np.asarray(result.eigenvectors, dtype=float)
    order = np.argsort(evals)[:n_requested]
    evals = evals[order]
    evecs = evecs[:, order]

    # Independent residual check: ||A x - w x|| per root.
    residual = 0.0
    sig = sigma(evecs)
    for k in range(evals.shape[0]):
        residual = max(
            residual,
            float(np.linalg.norm(sig[:, k] - evals[k] * evecs[:, k])),
        )
    return evals, evecs, residual


# ---------------------------------------------------------------------------
# Full Casida solver
# ---------------------------------------------------------------------------


def _build_casida_matrices(
    mo_energies: np.ndarray,
    eri_mo_ovov: np.ndarray,
    eri_mo_oovv: np.ndarray,
    n_occ: int,
    c_x: float,
    kernel_mo: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the full Casida A and B matrices.

    A_{ia,jb} = delta_ij delta_ab (eps_a - eps_i) + 2 (ia|jb) - c_x (ij|ab)
                + (ia|f_xc|jb)
    B_{ia,jb} = 2 (ia|jb) - c_x (ib|ja) + (ia|f_xc|jb)

    The adiabatic f_xc is a local, frequency-independent kernel, so the
    same (ia|f_xc|jb) matrix element enters both A and B (Casida 1995;
    hybrid composition per Bauernschmitt & Ahlrichs 1996).
    """
    n_virt = mo_energies.shape[0] - n_occ
    n_pair = n_occ * n_virt

    eps_occ = mo_energies[:n_occ]
    eps_virt = mo_energies[n_occ:]

    A = np.zeros((n_pair, n_pair))
    B = np.zeros((n_pair, n_pair))

    # Diagonal of A
    idx = 0
    for i in range(n_occ):
        for a in range(n_virt):
            A[idx, idx] = eps_virt[a] - eps_occ[i]
            idx += 1

    # eri_mo_ovov[i,a,j,b] = (ia|jb) -- correct convention
    # (ib|ja) = eri_mo_ovov[i,b,j,a]
    has_kernel = kernel_mo is not None

    idx_i = 0
    for i in range(n_occ):
        for a in range(n_virt):
            idx_j = 0
            for j in range(n_occ):
                for b in range(n_virt):
                    A[idx_i, idx_j] += (
                        2.0 * eri_mo_ovov[i, a, j, b] - c_x * eri_mo_oovv[i, j, a, b]
                    )
                    B[idx_i, idx_j] = (
                        2.0 * eri_mo_ovov[i, a, j, b] - c_x * eri_mo_ovov[i, b, j, a]
                    )
                    if has_kernel:
                        f_xc = kernel_mo[i, a, j, b]
                        A[idx_i, idx_j] += f_xc
                        B[idx_i, idx_j] += f_xc
                    idx_j += 1
            idx_i += 1

    return A, B


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _emit_tddft_citations(
    output: Optional[Union[str, os.PathLike]],
    basis: BasisSet,
    functional: Optional[str],
    *,
    tda: bool,
    hybrid_kernel: bool = False,
) -> None:
    """Best-effort: write ``{output}.bibtex`` + ``.references`` citation
    siblings for a TDDFT excitation calculation.

    Fires the linear-response TDDFT formalism papers -- Runge-Gross 1984 +
    Casida 1995 via ``uses_tddft`` -- plus Hirata-Head-Gordon 1999 via
    ``tddft_variant="tda"`` for the Tamm-Dancoff approximation and
    Bauernschmitt-Ahlrichs 1996 via ``tddft_hybrid_kernel`` when the
    response kernel carries a hybrid's exact-exchange admixture, alongside
    the XC functional (for the TDA-DFT / ALDA kernel) and the AO basis. The
    excitation solver runs no SCF iterations of its own, so the ground-state
    SCF accelerator (DIIS) is *not* credited here -- that citation belongs to
    the SCF that produced the input MOs and is emitted by the SCF runner.

    Non-fatal: a routing miss or writer failure is swallowed, mirroring the
    molecular/periodic runner behaviour, so it never tanks a finished
    excitation calculation. A ``None`` ``output`` is a no-op (the callers
    keep their historical "compute only, write nothing" default).
    """
    if output is None:
        return
    try:
        from .output.citations import emit_citations

        emit_citations(
            output,
            basis=getattr(basis, "name", None),
            functional=functional,
            uses_scf=False,
            uses_tddft=True,
            tddft_variant=("tda" if tda else None),
            tddft_hybrid_kernel=hybrid_kernel,
        )
    except Exception:
        # Citations are observability, not the load-bearing result.
        pass


def run_tddft_tda(
    molecule: Molecule,
    basis: BasisSet,
    mo_energies: np.ndarray,
    mo_coeff: np.ndarray,
    n_occ: int,
    *,
    n_states: int = 5,
    functional: Optional[str] = None,
    density_ao: Optional[np.ndarray] = None,
    output: Optional[Union[str, os.PathLike]] = None,
    grid_options: Optional[GridOptions] = None,
) -> TDDFTResult:
    """Compute vertical excitation energies via the Tamm-Dancoff
    approximation (CIS for HF, TDA-DFT for DFT).

    When ``functional`` and ``density_ao`` are both provided, the
    adiabatic LDA/GGA XC kernel (ALDA/AGGA) is computed on the DFT
    grid and added to the Casida A matrix.

    Parameters
    ----------
    molecule : Molecule
        The molecular system.
    basis : BasisSet
        AO basis set.
    mo_energies : np.ndarray
        Ground-state MO energies in Hartree, shape ``(n_basis,)``,
        sorted by energy.
    mo_coeff : np.ndarray
        Ground-state MO coefficients, shape ``(n_basis, n_basis)``,
        columns = MOs.
    n_occ : int
        Number of occupied MOs (RHF: n_electrons / 2; UHF: n_alpha).
    n_states : int
        Number of excited states to compute.
    grid_options : GridOptions or None
        XC response grid. Pass the reference SCF grid for consistency;
        None preserves the low-level grid default.
    functional : str or None
        XC functional of the ground-state reference; ``None`` for pure
        HF/CIS. Sets the exact-exchange coefficient of the response
        kernel (1 for HF, 0 for pure functionals, the admixture a for
        global hybrids). Range-separated hybrids raise
        ``NotImplementedError``.

    Returns
    -------
    TDDFTResult
        Container with excitation energies and properties.
    """
    n_basis = mo_coeff.shape[0]
    n_virt = n_basis - n_occ
    n_pair = n_occ * n_virt

    if n_pair == 0:
        raise ValueError("No virtual orbitals available for TDDFT (n_basis == n_occ).")

    # Exact-exchange admixture of the response kernel: 1 for HF/CIS,
    # 0 for pure functionals, a for global hybrids (RSH refused).
    c_x = _resolve_exact_exchange_fraction(functional)

    # AO to MO integral transformation (C++ kernel)
    eri_ao = compute_eri(basis)
    eri_mo_ovov, eri_mo_oovv = eri_ao_to_mo_blocks(
        np.asarray(eri_ao, dtype=float), mo_coeff, n_occ
    )
    del eri_ao  # full AO ERI tensor (O(n_basis^4)) — done after the MO transform

    # ALDA/AGGA XC kernel (only when functional + density provided)
    kernel_mo = None
    if functional is not None and density_ao is not None:
        kernel_mo = _compute_alda_kernel_mo(
            molecule,
            basis,
            functional,
            mo_coeff,
            density_ao,
            n_occ,
            grid_options=grid_options,
        )
    elif functional is not None and c_x != 0.0:
        _warn_hybrid_without_fxc(functional, c_x, "no density_ao provided")

    n_requested = min(n_states, n_pair)

    # Dense below the threshold -- it is faster there and reproduces the
    # pre-iterative numbers exactly.  Above it, forming A costs dim**2 doubles
    # on top of an eri_mo_ovov tensor of the same size and eigh is O(dim**3),
    # so iterate on the roots actually wanted instead.  Both paths solve the
    # same operator; tests/test_tddft_davidson.py pins them against each other.
    use_iterative = (
        n_pair > TDA_DAVIDSON_MIN_DIM
        and n_requested <= TDA_DAVIDSON_MAX_ROOT_FRACTION * n_pair
    )

    if use_iterative:
        sigma, diagonal = tda_sigma_and_diagonal(
            mo_energies,
            eri_mo_ovov,
            eri_mo_oovv,
            n_occ,
            c_x,
            kernel_mo=kernel_mo,
        )
        eigenvals, eigenvecs, residual = _solve_tda_iterative(
            sigma, diagonal, n_pair, n_requested
        )
        # The solver's own convergence flag is not usable for a partial
        # spectrum (#503), so act on the residual measured here instead.
        if residual > 1e-4:
            warnings.warn(
                f"TDA iterative solve returned a maximum root residual of "
                f"{residual:.2e}; excitation energies may be unconverged. "
                f"Request more than {TDA_DAVIDSON_MAX_ROOT_FRACTION:g} x "
                f"{n_pair} states to force the dense path.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        A = _build_tda_matrix(
            mo_energies,
            eri_mo_ovov,
            eri_mo_oovv,
            n_occ,
            c_x,
            kernel_mo=kernel_mo,
        )
        eigenvals, eigenvecs = np.linalg.eigh(A)
        # eigh sorts by eigenvalue ascending; take the first n_requested
        eigenvals = eigenvals[:n_requested]
        eigenvecs = eigenvecs[:, :n_requested]

    # Normalise eigenvectors to unity (they should already be, but
    # floating-point round-off can degrade this).
    for k in range(n_requested):
        norm = np.linalg.norm(eigenvecs[:, k])
        if norm > 0:
            eigenvecs[:, k] /= norm

    # Transition dipoles
    dipole_ao = compute_dipole(basis)

    result = TDDFTResult(
        n_states=n_requested,
        n_occ=n_occ,
        n_virt=n_virt,
        method="TDA",
        functional=functional,
    )

    for k in range(n_requested):
        omega = float(eigenvals[k])
        X = eigenvecs[:, k]

        # Ensure positive excitation energy
        if omega < 0:
            omega = abs(omega)

        td = _transition_dipole_from_amplitudes(X, n_occ, n_virt, mo_coeff, dipole_ao)
        f_osc = oscillator_strength(omega, td)
        dom = _dominant_amplitudes(X, n_occ, n_virt)

        ev = omega * 27.211386245988  # Hartree -> eV
        wl = (
            45.5633526 / omega if omega > 1e-12 else float("inf")
        )  # nm (from l[nm] = hc/w with constants)

        state = TDDFTState(
            index=k + 1,
            excitation_energy=omega,
            excitation_energy_ev=ev,
            wavelength_nm=wl,
            oscillator_strength=f_osc,
            transition_dipole=td,
            dominant_amplitudes=dom,
            excitation_vector=X.copy(),
        )
        result.states.append(state)

    _emit_tddft_citations(
        output,
        basis,
        functional,
        tda=True,
        hybrid_kernel=(functional is not None and c_x != 0.0),
    )
    return result


def run_tddft_casida(
    molecule: Molecule,
    basis: BasisSet,
    mo_energies: np.ndarray,
    mo_coeff: np.ndarray,
    n_occ: int,
    *,
    n_states: int = 5,
    functional: Optional[str] = None,
    density_ao: Optional[np.ndarray] = None,
    output: Optional[Union[str, os.PathLike]] = None,
    grid_options: Optional[GridOptions] = None,
) -> TDDFTResult:
    """Compute vertical excitation energies via the full Casida equation
    (linear-response TDDFT / TD-HF).

    When ``functional`` and ``density_ao`` are both provided, the
    adiabatic LDA/GGA XC kernel (ALDA/AGGA) is computed on the DFT
    grid and added to both the A and B matrices.

    Parameters
    ----------
    molecule : Molecule
        The molecular system.
    basis : BasisSet
        AO basis set.
    mo_energies : np.ndarray
        Ground-state MO energies in Hartree, shape ``(n_basis,)``.
    mo_coeff : np.ndarray
        Ground-state MO coefficients, shape ``(n_basis, n_basis)``,
        columns = MOs.
    n_occ : int
        Number of occupied MOs.
    n_states : int
        Number of excited states to compute.
    grid_options : GridOptions or None
        XC response grid. Pass the reference SCF grid for consistency;
        None preserves the low-level grid default.
    functional : str or None
        XC functional of the ground-state reference; ``None`` for pure
        HF/TDHF. Sets the exact-exchange coefficient of the response
        kernel (1 for HF, 0 for pure functionals, the admixture a for
        global hybrids). Range-separated hybrids raise
        ``NotImplementedError``.
    density_ao : np.ndarray or None
        Converged ground-state AO density matrix; required for the
        f_xc kernel (omitting it with a functional gives the bare /
        exchange-only response, not full TDDFT).

    Returns
    -------
    TDDFTResult
    """
    n_basis = mo_coeff.shape[0]
    n_virt = n_basis - n_occ
    n_pair = n_occ * n_virt

    if n_pair == 0:
        raise ValueError("No virtual orbitals available for TDDFT (n_basis == n_occ).")

    # Exact-exchange admixture of the response kernel: 1 for HF/TDHF,
    # 0 for pure functionals, a for global hybrids (RSH refused).
    c_x = _resolve_exact_exchange_fraction(functional)

    # AO to MO integral transformation (C++ kernel)
    eri_ao = compute_eri(basis)
    eri_mo_ovov, eri_mo_oovv = eri_ao_to_mo_blocks(
        np.asarray(eri_ao, dtype=float), mo_coeff, n_occ
    )
    del eri_ao  # full AO ERI tensor (O(n_basis^4)) — done after the MO transform

    # ALDA/AGGA XC kernel (only when functional + density provided)
    kernel_mo = None
    if functional is not None and density_ao is not None:
        kernel_mo = _compute_alda_kernel_mo(
            molecule,
            basis,
            functional,
            mo_coeff,
            density_ao,
            n_occ,
            grid_options=grid_options,
        )
    elif functional is not None and c_x != 0.0:
        _warn_hybrid_without_fxc(functional, c_x, "no density_ao provided")

    # Build A and B
    A, B = _build_casida_matrices(
        mo_energies,
        eri_mo_ovov,
        eri_mo_oovv,
        n_occ,
        c_x,
        kernel_mo=kernel_mo,
    )

    # Casida: (A - B)^{1/2} (A + B) (A - B)^{1/2} Z = w^2 Z
    A_minus_B = A - B
    A_plus_B = A + B

    # Square root of A-B (real symmetric positive-definite for stable
    # ground states)
    eig_amb, U_amb = np.linalg.eigh(A_minus_B)
    # Clamp negative eigenvalues to zero (shouldn't happen for stable SCF)
    eig_amb = np.maximum(eig_amb, 0.0)
    sqrt_amb_inv = np.diag(1.0 / np.sqrt(np.maximum(eig_amb, 1e-14)))
    sqrt_amb = np.diag(np.sqrt(eig_amb))

    A_minus_B_sqrt = U_amb @ sqrt_amb @ U_amb.T
    A_minus_B_sqrt_inv = U_amb @ sqrt_amb_inv @ U_amb.T

    # Effective Hamiltonian
    M = A_minus_B_sqrt @ A_plus_B @ A_minus_B_sqrt
    # Ensure symmetry
    M = 0.5 * (M + M.T)

    omega_sq, Z = np.linalg.eigh(M)
    omega_sq = np.maximum(omega_sq, 0.0)
    omega_vals = np.sqrt(omega_sq)

    # Sort by excitation energy
    idx_sort = np.argsort(omega_vals)
    n_requested = min(n_states, n_pair)
    omega_vals = omega_vals[idx_sort[:n_requested]]
    Z = Z[:, idx_sort[:n_requested]]

    # Recover excitation vectors: X + Y = (A-B)^{-1/2} Z . √w
    #                          X - Y = (A-B)^{1/2} Z / √w
    X_plus_Y_all = A_minus_B_sqrt_inv @ Z
    X_minus_Y_all = A_minus_B_sqrt @ Z

    # Transition dipoles
    dipole_ao = compute_dipole(basis)

    result = TDDFTResult(
        n_states=n_requested,
        n_occ=n_occ,
        n_virt=n_virt,
        method="Casida",
        functional=functional,
    )

    for k in range(n_requested):
        omega = float(omega_vals[k])
        XpY = X_plus_Y_all[:, k]
        # Normalise: |X+Y|^2 - |X-Y|^2 = 1 (excitation norm)
        # Use X+Y for transition properties
        norm_factor = np.sqrt(max(np.dot(XpY, XpY), 1e-14))
        if norm_factor > 0:
            XpY = XpY / norm_factor

        td = _transition_dipole_from_amplitudes(XpY, n_occ, n_virt, mo_coeff, dipole_ao)
        f_osc = oscillator_strength(omega, td)
        dom = _dominant_amplitudes(XpY, n_occ, n_virt)

        ev = omega * 27.211386245988
        wl = 45.5633526 / omega if omega > 1e-12 else float("inf")

        state = TDDFTState(
            index=k + 1,
            excitation_energy=omega,
            excitation_energy_ev=ev,
            wavelength_nm=wl,
            oscillator_strength=f_osc,
            transition_dipole=td,
            dominant_amplitudes=dom,
            excitation_vector=XpY.copy(),
        )
        result.states.append(state)

    _emit_tddft_citations(
        output,
        basis,
        functional,
        tda=False,
        hybrid_kernel=(functional is not None and c_x != 0.0),
    )
    return result


# ---------------------------------------------------------------------------
# UHF / UKS TDDFT
# ---------------------------------------------------------------------------


def _build_uhf_tda_matrix(
    mo_energies_a: np.ndarray,
    mo_energies_b: np.ndarray,
    mo_coeffs_a: np.ndarray,
    mo_coeffs_b: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
    eri_ao: np.ndarray,
    c_x: float,
    kernel_blocks: Optional[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = None,
) -> np.ndarray:
    """Build the UHF TDA matrix.

    A^{ss'}_{ia,jb} = d_{ss'} d_{ij} d_{ab} (e_a^s - e_i^s)
                     + (i_s a_s | j_{s'} b_{s'})
                     - c_x d_{ss'} (i_s j_s | a_s b_s)
    """
    n_virt_a = mo_energies_a.shape[0] - n_occ_a
    n_virt_b = mo_energies_b.shape[0] - n_occ_b
    n_pair_a = n_occ_a * n_virt_a
    n_pair_b = n_occ_b * n_virt_b
    n_total = n_pair_a + n_pair_b

    eps_occ_a = mo_energies_a[:n_occ_a]
    eps_virt_a = mo_energies_a[n_occ_a:]
    eps_occ_b = mo_energies_b[:n_occ_b]
    eps_virt_b = mo_energies_b[n_occ_b:]

    A = np.zeros((n_total, n_total))

    # Diagonal: orbital energy differences
    idx = 0
    for i in range(n_occ_a):
        for a in range(n_virt_a):
            A[idx, idx] = eps_virt_a[a] - eps_occ_a[i]
            idx += 1
    for i in range(n_occ_b):
        for a in range(n_virt_b):
            A[idx, idx] = eps_virt_b[a] - eps_occ_b[i]
            idx += 1

    off_a = 0
    off_b = n_pair_a

    # Transform ERIs (C++ kernel for ovov + oovv in one call)
    eri_aaaa, eri_aaaa_oovv = eri_ao_to_mo_blocks(
        np.asarray(eri_ao, dtype=float), mo_coeffs_a, n_occ_a
    )
    eri_bbbb, eri_bbbb_oovv = eri_ao_to_mo_blocks(
        np.asarray(eri_ao, dtype=float), mo_coeffs_b, n_occ_b
    )

    # Cross-spin: (i_a a_a | j_b b_b) via C++ kernel
    eri_cross = eri_ao_to_mo_cross(
        np.asarray(eri_ao, dtype=float),
        np.asarray(mo_coeffs_a, dtype=float),
        np.asarray(mo_coeffs_b, dtype=float),
        n_occ_a,
        n_occ_b,
    )

    # A_aa block: (i_a a_a | j_a b_a) - c_x (i_a j_a | a_a b_a)
    # ovov[i,a,j,b] -> reshape to (n_pair_a, n_pair_a)
    # oovv[i,j,a,b] -> transpose axes then reshape
    A_aa_ovov = eri_aaaa.reshape(n_pair_a, n_pair_a)
    A_aa_oovv = eri_aaaa_oovv.transpose(0, 2, 1, 3).reshape(n_pair_a, n_pair_a)
    A[off_a : off_a + n_pair_a, off_a : off_a + n_pair_a] += A_aa_ovov - c_x * A_aa_oovv

    # A_bb block
    A_bb_ovov = eri_bbbb.reshape(n_pair_b, n_pair_b)
    A_bb_oovv = eri_bbbb_oovv.transpose(0, 2, 1, 3).reshape(n_pair_b, n_pair_b)
    A[off_b : off_b + n_pair_b, off_b : off_b + n_pair_b] += A_bb_ovov - c_x * A_bb_oovv

    # A_ab cross-spin (no exchange for different spin)
    A_ab = eri_cross.reshape(n_pair_a, n_pair_b)
    A[off_a : off_a + n_pair_a, off_b : off_b + n_pair_b] += A_ab
    A[off_b : off_b + n_pair_b, off_a : off_a + n_pair_a] += A_ab.T

    if kernel_blocks is not None:
        K_aa, K_ab, K_ba, K_bb = kernel_blocks
        A[off_a : off_a + n_pair_a, off_a : off_a + n_pair_a] += K_aa
        A[off_a : off_a + n_pair_a, off_b : off_b + n_pair_b] += K_ab
        A[off_b : off_b + n_pair_b, off_a : off_a + n_pair_a] += K_ba
        A[off_b : off_b + n_pair_b, off_b : off_b + n_pair_b] += K_bb

    return A


def _build_uhf_casida_matrices(
    mo_energies_a: np.ndarray,
    mo_energies_b: np.ndarray,
    mo_coeffs_a: np.ndarray,
    mo_coeffs_b: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
    eri_ao: np.ndarray,
    c_x: float,
    kernel_blocks: Optional[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build unrestricted TDDFT/TDHF A and B matrices."""
    A = _build_uhf_tda_matrix(
        mo_energies_a,
        mo_energies_b,
        mo_coeffs_a,
        mo_coeffs_b,
        n_occ_a,
        n_occ_b,
        eri_ao,
        c_x,
        kernel_blocks=kernel_blocks,
    )

    n_virt_a = mo_energies_a.shape[0] - n_occ_a
    n_virt_b = mo_energies_b.shape[0] - n_occ_b
    n_pair_a = n_occ_a * n_virt_a
    n_pair_b = n_occ_b * n_virt_b
    B = np.zeros_like(A)
    off_b = n_pair_a

    eri_aaaa, _ = eri_ao_to_mo_blocks(
        np.asarray(eri_ao, dtype=float), mo_coeffs_a, n_occ_a
    )
    eri_bbbb, _ = eri_ao_to_mo_blocks(
        np.asarray(eri_ao, dtype=float), mo_coeffs_b, n_occ_b
    )
    eri_cross = eri_ao_to_mo_cross(
        np.asarray(eri_ao, dtype=float),
        np.asarray(mo_coeffs_a, dtype=float),
        np.asarray(mo_coeffs_b, dtype=float),
        n_occ_a,
        n_occ_b,
    )

    def _same_spin_b(ovov: np.ndarray) -> np.ndarray:
        n_occ, n_virt = ovov.shape[0], ovov.shape[1]
        shape = (n_occ * n_virt, n_occ * n_virt)
        direct = ovov.reshape(shape)
        exchange = ovov.transpose(0, 3, 2, 1).reshape(shape)
        return direct - c_x * exchange

    B[:n_pair_a, :n_pair_a] += _same_spin_b(eri_aaaa)
    B[off_b : off_b + n_pair_b, off_b : off_b + n_pair_b] += _same_spin_b(
        eri_bbbb
    )
    B_ab = eri_cross.reshape(n_pair_a, n_pair_b)
    B[:n_pair_a, off_b : off_b + n_pair_b] += B_ab
    B[off_b : off_b + n_pair_b, :n_pair_a] += B_ab.T

    if kernel_blocks is not None:
        K_aa, K_ab, K_ba, K_bb = kernel_blocks
        B[:n_pair_a, :n_pair_a] += K_aa
        B[:n_pair_a, off_b : off_b + n_pair_b] += K_ab
        B[off_b : off_b + n_pair_b, :n_pair_a] += K_ba
        B[off_b : off_b + n_pair_b, off_b : off_b + n_pair_b] += K_bb

    return A, B


def _compute_polarised_kernel_mo_uhf(
    molecule: Molecule,
    basis: BasisSet,
    functional_name: str,
    mo_coeffs_a: np.ndarray,
    mo_coeffs_b: np.ndarray,
    density_alpha_ao: np.ndarray,
    density_beta_ao: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
    *,
    grid_options: Optional[GridOptions] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spin-polarised ``f_xc`` blocks for UHF/UKS TDA.

    For each spin-channel perturbation ``(j_s b_s)``, apply the
    spin-polarised AO XC kernel to ``(delta D_alpha, delta D_beta)``
    and project the returned alpha/beta potentials into the response
    spin blocks. The returned matrices are flattened as
    ``(alpha,alpha)``, ``(alpha,beta)``, ``(beta,alpha)``,
    ``(beta,beta)`` in the same pair ordering used by
    :func:`_build_uhf_tda_matrix`.
    """
    n_basis = mo_coeffs_a.shape[0]
    n_virt_a = n_basis - n_occ_a
    n_virt_b = n_basis - n_occ_b
    n_pair_a = n_occ_a * n_virt_a
    n_pair_b = n_occ_b * n_virt_b

    if grid_options is None:
        grid_options = GridOptions()
    grid = build_grid(molecule, grid_options)
    from ._vibeqc_core import evaluate_ao_with_gradient

    chi, dchi_x, dchi_y, dchi_z = evaluate_ao_with_gradient(basis, grid.points)
    func = Functional(functional_name, 2)
    builder = make_polarised_xc_kernel_builder(
        func,
        grid,
        chi,
        dchi_x,
        dchi_y,
        dchi_z,
        np.asarray(density_alpha_ao, dtype=np.float64),
        np.asarray(density_beta_ao, dtype=np.float64),
    )

    C_occ_a = np.ascontiguousarray(mo_coeffs_a[:, :n_occ_a])
    C_virt_a = np.ascontiguousarray(mo_coeffs_a[:, n_occ_a:])
    C_occ_b = np.ascontiguousarray(mo_coeffs_b[:, :n_occ_b])
    C_virt_b = np.ascontiguousarray(mo_coeffs_b[:, n_occ_b:])

    zeros = np.zeros((n_basis, n_basis), dtype=np.float64)
    K_aa = np.zeros((n_pair_a, n_pair_a), dtype=np.float64)
    K_ab = np.zeros((n_pair_a, n_pair_b), dtype=np.float64)
    K_ba = np.zeros((n_pair_b, n_pair_a), dtype=np.float64)
    K_bb = np.zeros((n_pair_b, n_pair_b), dtype=np.float64)

    def _project_alpha(W: np.ndarray) -> np.ndarray:
        return (C_occ_a.T @ W @ C_virt_a).reshape(n_pair_a)

    def _project_beta(W: np.ndarray) -> np.ndarray:
        return (C_occ_b.T @ W @ C_virt_b).reshape(n_pair_b)

    for j in range(n_occ_a):
        c_j = C_occ_a[:, j]
        for b in range(n_virt_a):
            col = j * n_virt_a + b
            c_b = C_virt_a[:, b]
            dDa = np.outer(c_j, c_b) + np.outer(c_b, c_j)
            Wa, Wb = (np.asarray(x) for x in builder.apply(dDa, zeros))
            # libxc's spin-polarised second derivatives are with respect
            # to (rho_alpha, rho_beta). Under the closed-shell reduction,
            # f_total = 1/2 * (f_aa + f_ab); keep the same convention for
            # each spin block so identical alpha/beta references reproduce
            # the restricted TDDFT kernel.
            K_aa[:, col] = 0.5 * _project_alpha(Wa)
            K_ba[:, col] = 0.5 * _project_beta(Wb)

    for j in range(n_occ_b):
        c_j = C_occ_b[:, j]
        for b in range(n_virt_b):
            col = j * n_virt_b + b
            c_b = C_virt_b[:, b]
            dDb = np.outer(c_j, c_b) + np.outer(c_b, c_j)
            Wa, Wb = (np.asarray(x) for x in builder.apply(zeros, dDb))
            K_ab[:, col] = 0.5 * _project_alpha(Wa)
            K_bb[:, col] = 0.5 * _project_beta(Wb)

    return K_aa, K_ab, K_ba, K_bb


def _uhf_transition_dipole(
    X: np.ndarray,
    n_occ_a: int,
    n_virt_a: int,
    n_occ_b: int,
    n_virt_b: int,
    mo_coeffs_a: np.ndarray,
    mo_coeffs_b: np.ndarray,
    dipole_ao,
) -> np.ndarray:
    """Transition dipole moment for a UHF excitation vector."""
    n_pair_a = n_occ_a * n_virt_a
    X_a = X[:n_pair_a].reshape(n_occ_a, n_virt_a)
    X_b = X[n_pair_a:].reshape(n_occ_b, n_virt_b)

    C_occ_a = mo_coeffs_a[:, :n_occ_a]
    C_virt_a = mo_coeffs_a[:, n_occ_a:]
    C_occ_b = mo_coeffs_b[:, :n_occ_b]
    C_virt_b = mo_coeffs_b[:, n_occ_b:]

    P_a = C_occ_a @ X_a @ C_virt_a.T + C_virt_a @ X_a.T @ C_occ_a.T
    P_b = C_occ_b @ X_b @ C_virt_b.T + C_virt_b @ X_b.T @ C_occ_b.T
    P_ao = P_a + P_b

    mu = np.zeros(3)
    for d, comp in enumerate([dipole_ao.x, dipole_ao.y, dipole_ao.z]):
        mu[d] = np.trace(P_ao @ np.asarray(comp))
    return mu


def run_tddft_tda_uhf(
    molecule: Molecule,
    basis: BasisSet,
    mo_energies_a: np.ndarray,
    mo_energies_b: np.ndarray,
    mo_coeffs_a: np.ndarray,
    mo_coeffs_b: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
    *,
    n_states: int = 5,
    functional: Optional[str] = None,
    density_alpha_ao: Optional[np.ndarray] = None,
    density_beta_ao: Optional[np.ndarray] = None,
    output: Optional[Union[str, os.PathLike]] = None,
    grid_options: Optional[GridOptions] = None,
) -> TDDFTResult:
    """UHF Tamm-Dancoff approximation (UCIS for UHF).

    Parameters
    ----------
    molecule, basis : standard inputs
    mo_energies_a, mo_energies_b : alpha/beta MO energies
    mo_coeffs_a, mo_coeffs_b : alpha/beta MO coefficients
    n_occ_a, n_occ_b : occupied alpha/beta orbitals
    grid_options : GridOptions or None
        XC response grid. Pass the reference SCF grid for consistency;
        None preserves the low-level grid default.
    n_states : number of excited states
    functional : XC functional of the ground-state reference, or None
        for UHF/UCIS. Sets the exact-exchange coefficient of the
        same-spin exchange blocks (1 for HF, 0 for pure functionals,
        the admixture a for global hybrids; range-separated hybrids
        raise ``NotImplementedError``). When ``density_alpha_ao`` and
        ``density_beta_ao`` are provided, the spin-polarised adiabatic
        ``f_xc`` kernel is included for LDA/GGA functionals.
    """
    n_virt_a = mo_energies_a.shape[0] - n_occ_a
    n_virt_b = mo_energies_b.shape[0] - n_occ_b
    n_total = n_occ_a * n_virt_a + n_occ_b * n_virt_b

    if n_total == 0:
        raise ValueError("No virtual orbitals available for UHF TDDFT.")

    # Exact-exchange admixture of the response kernel: 1 for HF/UCIS,
    # 0 for pure functionals, a for global hybrids (RSH refused).
    c_x = _resolve_exact_exchange_fraction(functional)
    kernel_blocks = None
    if (
        functional is not None
        and density_alpha_ao is not None
        and density_beta_ao is not None
    ):
        kernel_blocks = _compute_polarised_kernel_mo_uhf(
            molecule,
            basis,
            functional,
            np.asarray(mo_coeffs_a, dtype=float),
            np.asarray(mo_coeffs_b, dtype=float),
            np.asarray(density_alpha_ao, dtype=float),
            np.asarray(density_beta_ao, dtype=float),
            n_occ_a,
            n_occ_b,
            grid_options=grid_options,
        )
    elif functional is not None and c_x != 0.0:
        _warn_hybrid_without_fxc(
            functional,
            c_x,
            "no spin-polarised density_ao provided",
        )
    eri_ao = compute_eri(basis)

    A = _build_uhf_tda_matrix(
        mo_energies_a,
        mo_energies_b,
        mo_coeffs_a,
        mo_coeffs_b,
        n_occ_a,
        n_occ_b,
        eri_ao,
        c_x,
        kernel_blocks=kernel_blocks,
    )

    n_requested = min(n_states, n_total)
    eigenvals, eigenvecs = np.linalg.eigh(A)
    eigenvals = eigenvals[:n_requested]
    eigenvecs = eigenvecs[:, :n_requested]

    for k in range(n_requested):
        norm = np.linalg.norm(eigenvecs[:, k])
        if norm > 0:
            eigenvecs[:, k] /= norm

    dipole_ao = compute_dipole(basis)

    result = TDDFTResult(
        n_states=n_requested,
        n_occ=n_occ_a + n_occ_b,
        n_virt=n_virt_a + n_virt_b,
        method="TDA-UHF",
        functional=functional,
    )

    for k in range(n_requested):
        omega = float(abs(eigenvals[k]))
        X = eigenvecs[:, k]
        td = _uhf_transition_dipole(
            X,
            n_occ_a,
            n_virt_a,
            n_occ_b,
            n_virt_b,
            mo_coeffs_a,
            mo_coeffs_b,
            dipole_ao,
        )
        f_osc = oscillator_strength(omega, td)

        ev = omega * 27.211386245988
        wl = 45.5633526 / omega if omega > 1e-12 else float("inf")

        state = TDDFTState(
            index=k + 1,
            excitation_energy=omega,
            excitation_energy_ev=ev,
            wavelength_nm=wl,
            oscillator_strength=f_osc,
            transition_dipole=td,
            dominant_amplitudes=[],
            excitation_vector=X.copy(),
        )
        result.states.append(state)

    _emit_tddft_citations(
        output,
        basis,
        functional,
        tda=True,
        hybrid_kernel=(functional is not None and c_x != 0.0),
    )
    return result


def run_tddft_casida_uhf(
    molecule: Molecule,
    basis: BasisSet,
    mo_energies_a: np.ndarray,
    mo_energies_b: np.ndarray,
    mo_coeffs_a: np.ndarray,
    mo_coeffs_b: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
    *,
    n_states: int = 5,
    functional: Optional[str] = None,
    density_alpha_ao: Optional[np.ndarray] = None,
    density_beta_ao: Optional[np.ndarray] = None,
    output: Optional[Union[str, os.PathLike]] = None,
    grid_options: Optional[GridOptions] = None,
) -> TDDFTResult:
    """Full unrestricted Casida TDDFT/TDHF for UHF/UKS references.

    ``grid_options`` selects the XC response grid; pass the reference SCF
    grid for consistency. None preserves the low-level grid default.
    """
    n_virt_a = mo_energies_a.shape[0] - n_occ_a
    n_virt_b = mo_energies_b.shape[0] - n_occ_b
    n_total = n_occ_a * n_virt_a + n_occ_b * n_virt_b
    if n_total == 0:
        raise ValueError("No virtual orbitals available for UHF TDDFT.")

    c_x = _resolve_exact_exchange_fraction(functional)
    kernel_blocks = None
    if (
        functional is not None
        and density_alpha_ao is not None
        and density_beta_ao is not None
    ):
        kernel_blocks = _compute_polarised_kernel_mo_uhf(
            molecule,
            basis,
            functional,
            np.asarray(mo_coeffs_a, dtype=float),
            np.asarray(mo_coeffs_b, dtype=float),
            np.asarray(density_alpha_ao, dtype=float),
            np.asarray(density_beta_ao, dtype=float),
            n_occ_a,
            n_occ_b,
            grid_options=grid_options,
        )
    elif functional is not None and c_x != 0.0:
        _warn_hybrid_without_fxc(
            functional,
            c_x,
            "no spin-polarised density_ao provided",
        )

    eri_ao = compute_eri(basis)
    A, B = _build_uhf_casida_matrices(
        mo_energies_a,
        mo_energies_b,
        mo_coeffs_a,
        mo_coeffs_b,
        n_occ_a,
        n_occ_b,
        eri_ao,
        c_x,
        kernel_blocks=kernel_blocks,
    )

    A_minus_B = A - B
    A_plus_B = A + B
    eig_amb, U_amb = np.linalg.eigh(A_minus_B)
    eig_amb = np.maximum(eig_amb, 0.0)
    sqrt_amb_inv = np.diag(1.0 / np.sqrt(np.maximum(eig_amb, 1e-14)))
    sqrt_amb = np.diag(np.sqrt(eig_amb))
    A_minus_B_sqrt = U_amb @ sqrt_amb @ U_amb.T
    A_minus_B_sqrt_inv = U_amb @ sqrt_amb_inv @ U_amb.T

    M = A_minus_B_sqrt @ A_plus_B @ A_minus_B_sqrt
    M = 0.5 * (M + M.T)
    omega_sq, Z = np.linalg.eigh(M)
    omega_vals = np.sqrt(np.maximum(omega_sq, 0.0))
    idx_sort = np.argsort(omega_vals)
    n_requested = min(n_states, n_total)
    omega_vals = omega_vals[idx_sort[:n_requested]]
    Z = Z[:, idx_sort[:n_requested]]
    X_plus_Y_all = A_minus_B_sqrt_inv @ Z

    dipole_ao = compute_dipole(basis)
    result = TDDFTResult(
        n_states=n_requested,
        n_occ=n_occ_a + n_occ_b,
        n_virt=n_virt_a + n_virt_b,
        method="Casida-UHF",
        functional=functional,
    )
    for k in range(n_requested):
        omega = float(omega_vals[k])
        XpY = X_plus_Y_all[:, k]
        norm = np.linalg.norm(XpY)
        if norm > 0:
            XpY = XpY / norm
        td = _uhf_transition_dipole(
            XpY,
            n_occ_a,
            n_virt_a,
            n_occ_b,
            n_virt_b,
            mo_coeffs_a,
            mo_coeffs_b,
            dipole_ao,
        )
        f_osc = oscillator_strength(omega, td)
        result.states.append(
            TDDFTState(
                index=k + 1,
                excitation_energy=omega,
                excitation_energy_ev=omega * 27.211386245988,
                wavelength_nm=45.5633526 / omega if omega > 1e-12 else float("inf"),
                oscillator_strength=f_osc,
                transition_dipole=td,
                dominant_amplitudes=[],
                excitation_vector=XpY.copy(),
            )
        )

    _emit_tddft_citations(
        output,
        basis,
        functional,
        tda=False,
        hybrid_kernel=(functional is not None and c_x != 0.0),
    )
    return result


# ---------------------------------------------------------------------------
# Periodic TDDFT (Gamma-only)
# ---------------------------------------------------------------------------


def run_tddft_tda_periodic(
    result,
    basis: BasisSet,
    *,
    n_occ: Optional[int] = None,
    n_states: int = 5,
    functional: Optional[str] = None,
    output: Optional[Union[str, os.PathLike]] = None,
) -> TDDFTResult:
    """TDDFT/TDA for periodic SCF results at the Gamma point.

    Extracts MO energies and coefficients from a converged periodic
    SCF result and runs the standard TDA/CIS solver. Works for
    vacuum-padded molecular cells and large supercells where
    periodic image interactions in the excitation kernel are
    negligible.

    Parameters
    ----------
    result
        A converged periodic SCF result with .mo_energies and
        .mo_coeffs attributes (PeriodicRHFResult, GpwScfResult).
    basis : BasisSet
        The AO basis set used for the SCF.
    n_occ : int or None
        Number of occupied orbitals. If None, must be provided.
    n_states : int
        Number of excited states.
    functional : str or None
        XC functional name for metadata.
    """
    mo_energies = np.asarray(result.mo_energies)
    mo_coeffs = np.asarray(result.mo_coeffs)

    if n_occ is None:
        raise ValueError("n_occ must be provided for periodic TDDFT.")

    # Build a minimal molecule wrapper
    try:
        from .molecule import Molecule

        mol = Molecule([], 0, 1)
    except Exception:
        mol = None

    return run_tddft_tda(
        mol,
        basis,
        mo_energies,
        mo_coeffs,
        n_occ,
        n_states=n_states,
        functional=functional,
        output=output,
    )


# ---------------------------------------------------------------------------
# Natural Transition Orbitals (NTO) analysis
# ---------------------------------------------------------------------------


@dataclass
class NTOResult:
    """Natural Transition Orbital analysis for a single excited state.

    Attributes
    ----------
    hole_weights : np.ndarray
        Singular values (weights) of the hole NTOs, sorted descending.
    particle_weights : np.ndarray
        Singular values of the particle NTOs (same as hole_weights for
        a full SVD; equal by construction).
    hole_orbitals_ao : np.ndarray
        Hole NTO coefficients in the AO basis, shape ``(n_basis, n_occ)``.
        Each column is a hole NTO; columns are ordered by decreasing weight.
    particle_orbitals_ao : np.ndarray
        Particle NTO coefficients in the AO basis,
        shape ``(n_basis, n_virt)``.
    dominant_pair : tuple
        ``(weight, hole_idx, particle_idx)`` for the dominant NTO pair.
    """

    hole_weights: np.ndarray
    particle_weights: np.ndarray
    hole_orbitals_ao: np.ndarray
    particle_orbitals_ao: np.ndarray
    dominant_pair: Tuple[float, int, int]


def compute_nto(
    state: TDDFTState,
    mo_coeff: np.ndarray,
    n_occ: int,
) -> NTOResult:
    """Compute Natural Transition Orbitals for a TDDFT excited state.

    Performs an SVD on the excitation amplitude matrix
    ``X[ia] = excitation_vector.reshape(n_occ, n_virt)``.
    The resulting hole/particle NTO pairs provide the most compact
    orbital representation of the excitation.

    Reference: Martin, R. L. *J. Chem. Phys.* **118**, 4775 (2003).

    Parameters
    ----------
    state : TDDFTState
        A single excited state from a TDDFT calculation.
    mo_coeff : np.ndarray
        Ground-state MO coefficient matrix, shape ``(n_basis, n_basis)``.
    n_occ : int
        Number of occupied orbitals.

    Returns
    -------
    NTOResult
    """
    n_basis = mo_coeff.shape[0]
    n_virt = n_basis - n_occ
    X = np.asarray(state.excitation_vector, dtype=float)

    if X.size != n_occ * n_virt:
        raise ValueError(
            f"Excitation vector size {X.size} does not match "
            f"n_occ * n_virt = {n_occ} * {n_virt} = {n_occ * n_virt}"
        )

    # Reshape to (n_occ, n_virt) and perform SVD
    X_ia = X.reshape(n_occ, n_virt)
    U, s, Vt = np.linalg.svd(X_ia, full_matrices=False)
    # s has length min(n_occ, n_virt)
    # U: (n_occ, k), Vt: (k, n_virt) where k = min(n_occ, n_virt)

    C_occ = np.asarray(mo_coeff[:, :n_occ])
    C_virt = np.asarray(mo_coeff[:, n_occ:])

    # Hole NTOs in AO basis: C_occ @ U  -> (n_basis, k)
    hole_ao = C_occ @ U
    # Particle NTOs in AO basis: C_virt @ Vt.T -> (n_basis, k)
    particle_ao = C_virt @ Vt.T

    # Dominant pair (largest singular value)
    if len(s) > 0:
        dominant = (float(s[0]), 0, 0)
    else:
        dominant = (0.0, -1, -1)

    return NTOResult(
        hole_weights=s.copy(),
        particle_weights=s.copy(),
        hole_orbitals_ao=hole_ao,
        particle_orbitals_ao=particle_ao,
        dominant_pair=dominant,
    )


def compute_nto_uhf(
    state: TDDFTState,
    mo_coeffs_a: np.ndarray,
    mo_coeffs_b: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
) -> dict[str, NTOResult]:
    """Compute NTOs for a UHF/TDA excited state.

    The UHF excitation vector has two blocks: first n_occ_a·n_virt_a
    alpha excitations, then n_occ_b·n_virt_b beta excitations.
    SVD is applied to each block separately, producing alpha and beta
    NTOs in the AO basis.

    Returns
    -------
    dict with keys ``"alpha"`` and ``"beta"``, each an :class:`NTOResult`.
    """
    n_pair_a = n_occ_a * (mo_coeffs_a.shape[0] - n_occ_a)
    n_pair_b = n_occ_b * (mo_coeffs_b.shape[0] - n_occ_b)
    X = np.asarray(state.excitation_vector, dtype=float)

    if X.size != n_pair_a + n_pair_b:
        raise ValueError(
            f"Excitation vector size {X.size} does not match "
            f"n_pair_a + n_pair_b = {n_pair_a} + {n_pair_b}"
        )

    X_a = X[:n_pair_a]
    X_b = X[n_pair_a:]

    # Alpha NTOs
    n_virt_a = mo_coeffs_a.shape[0] - n_occ_a
    X_a_ia = X_a.reshape(n_occ_a, n_virt_a)
    U_a, s_a, Vt_a = np.linalg.svd(X_a_ia, full_matrices=False)
    C_occ_a = np.asarray(mo_coeffs_a[:, :n_occ_a])
    C_virt_a = np.asarray(mo_coeffs_a[:, n_occ_a:])
    hole_a_ao = C_occ_a @ U_a
    particle_a_ao = C_virt_a @ Vt_a.T
    dom_a = (float(s_a[0]), 0, 0) if len(s_a) > 0 else (0.0, -1, -1)
    nto_a = NTOResult(
        hole_weights=s_a.copy(),
        particle_weights=s_a.copy(),
        hole_orbitals_ao=hole_a_ao,
        particle_orbitals_ao=particle_a_ao,
        dominant_pair=dom_a,
    )

    # Beta NTOs
    n_virt_b = mo_coeffs_b.shape[0] - n_occ_b
    X_b_ia = X_b.reshape(n_occ_b, n_virt_b)
    U_b, s_b, Vt_b = np.linalg.svd(X_b_ia, full_matrices=False)
    C_occ_b = np.asarray(mo_coeffs_b[:, :n_occ_b])
    C_virt_b = np.asarray(mo_coeffs_b[:, n_occ_b:])
    hole_b_ao = C_occ_b @ U_b
    particle_b_ao = C_virt_b @ Vt_b.T
    dom_b = (float(s_b[0]), 0, 0) if len(s_b) > 0 else (0.0, -1, -1)

    nto_b = NTOResult(
        hole_weights=s_b.copy(),
        particle_weights=s_b.copy(),
        hole_orbitals_ao=hole_b_ao,
        particle_orbitals_ao=particle_b_ao,
        dominant_pair=dom_b,
    )

    return {"alpha": nto_a, "beta": nto_b}


# ---------------------------------------------------------------------------
# ROHF TDA (restricted open-shell HF)
# ---------------------------------------------------------------------------


def run_tddft_tda_rohf(
    molecule: Molecule,
    basis: BasisSet,
    mo_energies: np.ndarray,
    mo_coeff: np.ndarray,
    n_docc: int,
    n_socc: int,
    *,
    n_states: int = 5,
    output: Optional[Union[str, os.PathLike]] = None,
) -> TDDFTResult:
    """TDA/CIS for ROHF reference.

    Only doubly-occupied → virtual excitations are included in this
    first implementation.  Singly-occupied orbital (SOMO) excitations
    require spin-adaptation and are deferred.

    Parameters
    ----------
    molecule, basis : standard
    mo_energies : ROHF MO energies (same for alpha and beta)
    mo_coeff : ROHF MO coefficients (spatial orbitals)
    n_docc : number of doubly-occupied orbitals
    n_socc : number of singly-occupied orbitals (alpha spin)
    n_states : excited states requested
    """
    n_basis = mo_coeff.shape[0]
    n_virt = n_basis - n_docc - n_socc
    n_pair = n_docc * n_virt

    if n_pair == 0:
        raise ValueError("No doubly-occ → virtual excitations available for ROHF TDA")

    # Compact MO space: skip singly-occupied orbitals.
    # douc: [0, n_docc), virt: [n_docc+n_socc, n_basis)
    _idx = list(range(n_docc)) + list(range(n_docc + n_socc, n_basis))
    _mo_e = np.asarray(mo_energies)[_idx]
    _mo_c = np.asarray(mo_coeff)[:, _idx]
    _n_occ = n_docc
    _n_virt = n_virt

    # ROHF CIS: same A-matrix as RHF for doubly-occ → virt.
    c_x = 1.0

    eri_ao = compute_eri(basis)
    eri_mo_ovov, eri_mo_oovv = eri_ao_to_mo_blocks(
        np.asarray(eri_ao, dtype=float),
        _mo_c,
        _n_occ,
    )

    A = _build_tda_matrix(
        _mo_e,
        eri_mo_ovov,
        eri_mo_oovv,
        _n_occ,
        c_x,
        kernel_mo=None,
    )

    n_requested = min(n_states, n_pair)
    eigenvals, eigenvecs = np.linalg.eigh(A)
    eigenvals = eigenvals[:n_requested]
    eigenvecs = eigenvecs[:, :n_requested]

    for k in range(n_requested):
        norm = np.linalg.norm(eigenvecs[:, k])
        if norm > 0:
            eigenvecs[:, k] /= norm

    # Transition dipoles use the compact MO coefficient subset
    dipole_ao = compute_dipole(basis)
    _dip_c = np.asarray(
        [
            np.asarray(dipole_ao.x)[:, _idx][_idx, :],
            np.asarray(dipole_ao.y)[:, _idx][_idx, :],
            np.asarray(dipole_ao.z)[:, _idx][_idx, :],
        ]
    )
    _mo_c_occ = _mo_c[:, :_n_occ]
    _mo_c_virt = _mo_c[:, _n_occ:]

    result = TDDFTResult(
        n_states=n_requested,
        n_occ=n_docc,
        n_virt=n_virt,
        method="TDA-ROHF",
        functional=None,
    )

    for k in range(n_requested):
        omega = float(abs(eigenvals[k]))
        X = eigenvecs[:, k]
        # Transition density matrix in compact MO basis
        X_ia = X.reshape(_n_occ, _n_virt)
        P_ao = _mo_c_occ @ X_ia @ _mo_c_virt.T
        td = np.array(
            [
                float(np.sum(P_ao * np.asarray(dipole_ao.x))),
                float(np.sum(P_ao * np.asarray(dipole_ao.y))),
                float(np.sum(P_ao * np.asarray(dipole_ao.z))),
            ]
        )
        f_osc = oscillator_strength(omega, td)
        dom = _dominant_amplitudes(X, _n_occ, _n_virt)
        ev = omega * 27.211386245988
        wl = 45.5633526 / omega if omega > 1e-12 else float("inf")

        state = TDDFTState(
            index=k + 1,
            excitation_energy=omega,
            excitation_energy_ev=ev,
            wavelength_nm=wl,
            oscillator_strength=f_osc,
            transition_dipole=td,
            dominant_amplitudes=dom,
            excitation_vector=X.copy(),
        )
        result.states.append(state)

    return result
