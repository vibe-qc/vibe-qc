"""Analytic state-specific RHF-CASSCF nuclear gradient.

Implements the CASSCF energy gradient dE/dR by contracting the effective
AO 1- and 2-particle density matrices with AO derivative integrals (via
the existing C++ gradient kernels), plus an overlap-gradient correction
from the orthonormalization response of the MO coefficients.

The gradient is the plain first derivative of a fully variational energy:

  dE/dR = tr(D . dh/dR) + 1/2 tr(Γ . dg/dR) - tr(W . dS/dR) + dE_nuc/dR

with D and Γ the CASSCF one- and two-particle densities and W = C F C^T
the energy-weighted density built from the (non-symmetric) MCSCF
generalized Fock matrix.  A converged state-specific CASSCF is stationary
with respect to every orbital rotation and every CI coefficient, so by
the 2n+1 rule the first derivative carries NO wavefunction-response term
-- there is no z-vector to solve for.  (Helgaker, Jorgensen & Olsen,
"Molecular Electronic-Structure Theory", Sec. 10.2 and Sec. 12.5; the
Handy-Schaefer z-vector of Sec. 12.5 / J. Chem. Phys. 81, 5031 (1984)
enters only for NON-variational energies such as CASPT2 / NEVPT2, whose
gradient modules reuse the orbital-response helpers below.)

Validation, all against a central finite difference of the converged
CASSCF energy (step 1e-3, in-repo oracle): the full vector agrees to
2.2e-7 Ha/bohr on H2/6-31G CAS(2,2) and 2.5e-7 on H2/cc-pVDZ (R = 1.4
bohr), and to 1.1e-6 on N2/cc-pVDZ CAS(6,6) at R = 2.074 bohr, where the
transverse components are 1.1e-16 and the residual is FD truncation
(GitLab #516).

History.  An earlier revision described this path as an "87 % z-vector-
free approximation" and offered ``compute_wz=True``, a CP-MCSCF "W^z
correction" assembled from a single reference z-vector contracted with
every column of the perturbed orbital gradient.  That term is a phantom
-- the missing 12 % it was written to recover was a defect in the
energy-weighted density that was fixed separately (2026-06-19) -- and
its single-reference contraction produced a spurious x-component on
linear molecules with p/d functions (0.357 Ha/bohr on H2/cc-pVDZ) and
a sign flip on H2/6-31G.  ``compute_wz=True`` is now an accepted no-op
alias of the exact analytic path (GitLab #516).

References
----------
Helgaker, Jorgensen & Olsen, "Molecular Electronic-Structure Theory",
  Sec. 10.2 (variational energies: no response in the first derivative)
  and Sec. 12.5 (MCSCF gradients; the z-vector for non-variational cases).
Celani & Werner, J. Chem. Phys. 119, 5044 (2003) -- CASSCF analytic
  gradients with density fitting.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import eigh, expm

from .._vibeqc_core import (
    Atom,
    BasisSet,
    Molecule,
)
from .._vibeqc_core import (
    compute_eri as _compute_eri,
)
from .._vibeqc_core import (
    compute_kinetic as _compute_kinetic,
)
from .._vibeqc_core import (
    compute_nuclear as _compute_nuclear,
)
from .._vibeqc_core import (
    compute_overlap as _compute_overlap,
)
from .._vibeqc_core import (
    nuclear_repulsion_gradient as _nuc_rep_grad,
)
from .._vibeqc_core import (
    one_electron_gradient_contribution as _one_el_grad,
)
from .._vibeqc_core import (
    overlap_gradient_contribution as _overlap_grad,
)
from .._vibeqc_core import (
    two_electron_gradient_casscf as _two_el_grad_casscf,
)


def _compute_casscf_gradient_numerical(
    mol,
    basis,
    C_mo: np.ndarray,
    h1e_cas: np.ndarray,
    h2e_cas: np.ndarray,
    n_core: int,
    n_act: int,
    fd_eps: float = 0.005,
) -> np.ndarray:
    """Full CASSCF gradient by numerical energy FD.

    Runs a complete CASSCF optimisation at each displaced geometry
    (R_a ± fd_eps) and central-differences the total energy.  This is
    exact within FD error and is the production fallback when the
    analytic W^z correction is not available or unreliable.

    Returns (n_atoms, 3) gradient in Hartree/bohr.
    """
    n_atoms = len(mol.atoms)
    atoms_list = list(mol.atoms)
    grad = np.zeros((n_atoms, 3))

    for a in range(n_atoms):
        for c in range(3):
            xyz_p = np.array([at.xyz for at in atoms_list], dtype=float)
            xyz_m = np.array([at.xyz for at in atoms_list], dtype=float)
            xyz_p[a, c] += fd_eps
            xyz_m[a, c] -= fd_eps

            for sign, xyz in [(+1, xyz_p), (-1, xyz_m)]:
                mol_d = Molecule(
                    [Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz)]
                )
                basis_d = BasisSet(mol_d, basis.name)
                from ..solvers import build_hamiltonian_mo, get_hf_orbital_provider
                from ..solvers._casscf import casscf

                C_hf_d = get_hf_orbital_provider(mol_d, basis_d)
                H_d = build_hamiltonian_mo(mol_d, basis_d, C_hf_d)
                sc_d = casscf(
                    H_d.h1e,
                    H_d.h2e,
                    n_act,
                    n_act,
                    n_core=n_core,
                    nuclear_repulsion=H_d.nuclear_repulsion,
                )
                if sign == +1:
                    E_p = sc_d.e_total
                else:
                    E_m = sc_d.e_total
            grad[a, c] = (E_p - E_m) / (2.0 * fd_eps)

    return grad


def compute_casscf_gradient(
    mol,
    basis,
    C_mo: np.ndarray,
    h1e_cas: np.ndarray,
    h2e_cas: np.ndarray,
    n_core: int,
    n_active_orb: int,
    *,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    compute_wz: bool | str = False,
    fd_eps_z: float = 1e-4,
    determinants: list | None = None,
    ci_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic CASSCF nuclear gradient.

    Builds the AO one- and two-particle density matrices from the
    CASSCF solution, then contracts with the AO derivative integrals
    using the existing C++ analytic-gradient kernels.

    Parameters
    ----------
    mol : Molecule
        Molecular geometry.
    basis : BasisSet
        AO basis set.
    C_mo : (n_ao, n_mo) ndarray
        AO->MO coefficient matrix (columns = MOs).  Must be the
        converged CASSCF basis (the cumulative rotation applied).
    h1e_cas : (n_mo, n_mo) ndarray
        One-electron MO integrals in the converged CASSCF basis.
    h2e_cas : (n_mo, n_mo, n_mo, n_mo) ndarray
        Two-electron MO integrals (physicist's g) in converged basis.
    n_core : int
        Number of doubly-occupied inactive (core) orbitals.
    n_active_orb : int
        Number of active spatial orbitals.
    rdm1 : (n_active_orb, n_active_orb) ndarray
        Active-space 1-RDM from CASCI (g_tu).
    rdm2 : (n_active_orb, n_active_orb, n_active_orb, n_active_orb) ndarray
        Active-space 2-RDM from CASCI (Γ_tuvw).
    compute_wz : bool or "numerical"
        ``False`` (default) and ``True`` both return the exact analytic
        gradient: a converged CASSCF is fully variational, so there is no
        response ("W^z") term to add, and the former experimental
        ``True`` path is retired (it produced a spurious x-component on
        linear molecules -- GitLab #516). ``True`` is accepted for
        backward compatibility and emits a :class:`FutureWarning`.
        ``"numerical"`` replaces the analytic result by a central finite
        difference of the re-converged CASSCF energy (the in-repo oracle;
        expensive).
    fd_eps_z : float
        Finite-difference step for ``compute_wz="numerical"``.
    determinants, ci_coeffs : optional
        Accepted for backward compatibility; the analytic gradient does
        not need them.

    Notes
    -----
    Manually supplied ECP-derived orbitals, integrals, or reference data
    paired with an all-electron-named basis are unsupported.  This API can
    reject an ECP attached to ``basis``, but cannot infer the Hamiltonian
    provenance of the supplied arrays.

    Returns
    -------
    grad : (n_atoms, 3) ndarray
        Nuclear gradient in Hartree/bohr.
    """
    from ..ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        mol,
        basis,
        route="compute_casscf_gradient",
    )
    nb = C_mo.shape[0]
    nmo = C_mo.shape[1]
    n_act = n_active_orb
    n_act_total = n_core + n_act  # nonzero MO index range

    # ---- 1. One-particle density in MO basis ----
    # Core: 2.I (doubly occupied)
    # Active: g_tu (from CASCI)
    # Virtual: 0
    # Mixed: 0 (z-vector-free approximation)
    D_mo = np.zeros((nmo, nmo))
    D_mo[:n_core, :n_core] = np.diag(np.full(n_core, 2.0))
    D_mo[n_core:n_act_total, n_core:n_act_total] = rdm1

    # Transform to AO: D_AO = C D_MO C^T
    D_ao = C_mo @ D_mo @ C_mo.T

    # ---- 2. Two-particle density in MO basis ----
    # Eq. 12.5.8-12.5.10 of Helgaker, Jorgensen & Olsen.
    # Uses the shared _build_full_gamma_mo helper (also used by CI coupling).
    gamma_mo = _build_full_gamma_mo(rdm1, rdm2, n_core, n_act, nmo)

    # ---- 3. Transform Γ_MO -> Γ_AO (4-index) ----
    # Flattened row-major: idx = ((mu*nb + nu)*nb + lam)*nb + sig
    gamma_ao_flat = _transform_4index_mo_to_ao_flat(gamma_mo, C_mo, n_act_total)

    # ---- 3b. Reorder axes for the C++ kernel convention ----
    # The C++ kernel contracts gamma(mu,ν,l,s) with d(muν|ls)/dR.
    # The libint derivative engine uses l-canonical shell reordering
    # which permutes (mu,ν) and (l,s).  For RHF, gamma has 8-fold
    # symmetry (from D⊗D), but the CASSCF 2-RDM has only 2-fold
    # symmetry (Γ_pqrs = Γ_rspq).  We must symmetrize gamma over
    # all 8 ERI-permutation-equivalent index orderings.
    gamma_ao_4d = gamma_ao_flat.reshape(nb, nb, nb, nb)
    # 8-fold symmetrization: average over (muν) swap, (ls) swap, and bra-ket swap
    gamma_sym = gamma_ao_4d.copy()
    gamma_sym += gamma_ao_4d.transpose(1, 0, 2, 3)  # swap mu,ν
    gamma_sym += gamma_ao_4d.transpose(0, 1, 3, 2)  # swap l,s
    gamma_sym += gamma_ao_4d.transpose(1, 0, 3, 2)  # swap both
    gamma_sym += gamma_ao_4d.transpose(2, 3, 0, 1)  # bra-ket swap
    gamma_sym += gamma_ao_4d.transpose(2, 3, 1, 0)  # bra-ket + swap muν
    gamma_sym += gamma_ao_4d.transpose(3, 2, 0, 1)  # bra-ket + swap ls
    gamma_sym += gamma_ao_4d.transpose(3, 2, 1, 0)  # bra-ket + swap both
    gamma_ao_flat = (gamma_sym / 8.0).ravel()

    # ---- 3c. Scale by 1/2 for the physical 2-electron energy prefactor ----
    # Eq. (12.5.7) of Helgaker, Jorgensen & Olsen:
    #   dE_2e/dR = S (1/2).Γ_muνls . d(muν|ls)/dR
    # The C++ two_electron_gradient_casscf sums gamma . dERI/dR with no extra
    # prefactor (same convention as the RHF kernel where gamma_RHF already
    # carries 0.5.D⊗D).  We supply gamma = 0.5.Γ_AO so the contraction
    # gives the correct physical gradient.
    # Validated: frozen-response FD matches to 2.5e-7 Ha/bohr on H2/6-31G
    # CAS(2,2) n_core=0 (2026-06-18).
    gamma_ao_flat *= 0.5

    # ---- 4. Gradient assembly ----
    # Nuclear repulsion gradient: dE_nuc/dR (returned by C++ as dE/dR)
    grad = np.asarray(_nuc_rep_grad(mol), dtype=float)

    # One-electron gradient: tr(D_AO . d(T+V_ne)/dR)
    grad += np.asarray(_one_el_grad(basis, mol, D_ao), dtype=float)

    # Two-electron gradient: (1/2).tr(Γ_AO . dg/dR), the 1/2 is in gamma_ao_flat
    grad += np.asarray(_two_el_grad_casscf(basis, mol, gamma_ao_flat), dtype=float)

    # ---- 5. Overlap gradient term (orthonormality constraint) ----
    # The density-contracted pieces above use C_conv at the reference
    # geometry.  The overlap change dS/dR forces C to stay orthonormal,
    # contributing -tr(W . dS/dR) with W = C F C^T the energy-weighted
    # density of the MCSCF generalized Fock matrix (Helgaker, Jorgensen &
    # Olsen Eq. 12.5.11 form).  This completes the gradient: a converged
    # CASSCF is stationary in every orbital and CI parameter, so the
    # first derivative has no wavefunction-response term (2n+1 rule).
    W_unrelaxed = _compute_w_unrelaxed(
        mol,
        basis,
        C_mo,
        D_mo,
        gamma_mo,
        n_act_total,
        n_core=n_core,
        n_act=n_active_orb,
        n_act_elec=int(round(np.trace(rdm1))),
        ms2=mol.multiplicity - 1,
        h1e_cas=h1e_cas,
        h2e_cas=h2e_cas,
    )
    grad += np.asarray(_overlap_grad(basis, mol, W_unrelaxed), dtype=float)

    # ---- 6. Optional finite-difference oracle ----
    # compute_wz="numerical" replaces the analytic vector by a central FD
    # of the re-converged CASSCF energy.  compute_wz=True used to add an
    # experimental CP-MCSCF "W^z correction" here; for a variational
    # CASSCF that term is identically zero, and the implementation
    # (one reference z-vector contracted with every perturbed-gradient
    # column) produced a spurious x-component on linear molecules with
    # p/d functions -- 0.357 Ha/bohr on H2/cc-pVDZ, against a true
    # transverse gradient of zero -- while the analytic vector above
    # matches the FD oracle to ~2e-7 Ha/bohr (GitLab #516).  True is
    # therefore an accepted no-op alias of the analytic path.
    if compute_wz == "numerical":
        grad = _compute_casscf_gradient_numerical(
            mol, basis, C_mo, h1e_cas, h2e_cas, n_core, n_act, fd_eps_z
        )
        return grad
    if compute_wz is True:
        warnings.warn(
            "CASSCFOptions(compute_wz=True) is obsolete: the analytic "
            "CASSCF gradient is already the complete derivative of a "
            "variational energy (no z-vector term exists), and the former "
            "experimental W^z correction was retired for producing "
            "spurious components (GitLab #516). Returning the analytic "
            "gradient; drop the flag, or use compute_wz=\"numerical\" "
            "for a finite-difference cross-check.",
            FutureWarning,
            stacklevel=2,
        )

    return grad


def _transform_4index_mo_to_ao_flat(
    gamma_mo: np.ndarray,
    C: np.ndarray,
    n_nonzero: int,
) -> np.ndarray:
    """Transform a 4-index MO tensor to flat AO basis.

    Only the first ``n_nonzero`` MO indices are nonzero -- the
    transformation skips virtual orbitals.  Returns a flat row-major
    vector of length nb^4 for the C++ gradient kernel.

    Parameters
    ----------
    gamma_mo : (nmo, nmo, nmo, nmo) ndarray
        4-index tensor in MO basis.  Nonzero only in [0:n_nonzero).
    C : (nb, nmo) ndarray
        MO coefficients (columns = MOs).
    n_nonzero : int
        Number of MO indices that may be nonzero.

    Returns
    -------
    gamma_ao_flat : (nb^4,) ndarray, row-major
        idx = ((mu*nb + nu)*nb + lam)*nb + sig
    """
    nb = C.shape[0]
    # Use vectorised einsum for the full 4-index transform.
    # gamma_ao[mu,nu,lam,sig] = sum_{pqrs} C_mu,p C_nu,q C_lam,r C_sig,s * gamma_mo[p,q,r,s]
    # This is the standard linear transform preserving the tensor structure.
    C_act = C[:, :n_nonzero]
    gamma_act = gamma_mo[:n_nonzero, :n_nonzero, :n_nonzero, :n_nonzero]
    gamma_ao = np.einsum(
        "ap,bq,cr,ds,pqrs->abcd",
        C_act,
        C_act,
        C_act,
        C_act,
        gamma_act,
        optimize=True,
    )
    return np.ascontiguousarray(gamma_ao.ravel())


# ---------------------------------------------------------------------------
#  W_unrelaxed -- overlap gradient correction
# ---------------------------------------------------------------------------
def _compute_w_unrelaxed(
    mol,
    basis,
    C_mo: np.ndarray,
    D_mo: np.ndarray,
    gamma_mo: np.ndarray,
    n_nonzero: int,
    *,
    n_core: int = 0,
    n_act: int = 0,
    n_act_elec: int = 0,
    ms2: int = 0,
    h1e_cas: np.ndarray | None = None,
    h2e_cas: np.ndarray | None = None,
    fd_eps: float = 1e-6,
) -> np.ndarray:
    """Compute W_unrelaxed = -dE/dS.

    Uses the analytical formula W = C.F.C^T when h1e_cas/h2e_cas are
    provided (F is the non-symmetric MCSCF generalized Fock).  Falls
    back to numerical FD via CASCI re-solves at perturbed overlap.

    Returns W in the AO basis (nb, nb).
    """
    if h1e_cas is not None and h2e_cas is not None and n_act > 0:
        nmo = C_mo.shape[1]
        pairs = _nonredundant_pairs_local(n_core, n_act, nmo)
        eri_chem = h2e_cas.transpose(0, 2, 1, 3)
        dm1_a = D_mo[n_core : n_core + n_act, n_core : n_core + n_act]
        dm2_a = gamma_mo[
            n_core : n_core + n_act,
            n_core : n_core + n_act,
            n_core : n_core + n_act,
            n_core : n_core + n_act,
        ]
        _grad, F, _Favg = _build_casscf_fock_and_gradient(
            h1e_cas,
            eri_chem,
            dm1_a,
            dm2_a,
            n_core,
            n_act,
            nmo,
            pairs,
        )
        return C_mo @ F @ C_mo.T

    nb = C_mo.shape[0]
    nmo = C_mo.shape[1]

    # Reference AO integrals
    T_ref = np.asarray(_compute_kinetic(basis))
    V_ref = np.asarray(_compute_nuclear(basis, mol))
    h_ao = T_ref + V_ref
    g_ao = np.asarray(_compute_eri(basis))  # chemist's (mu nu | lam sig)
    S_ref = np.asarray(_compute_overlap(basis))

    # Energy at perturbed overlap: re-orthonormalize MOs and re-solve CASCI.
    # The RDM transformation approach leaks density into virtual orbitals
    # that are inaccessible in CAS; re-solving CASCI preserves the active-
    # space structure and gives the physically correct frozen-CI energy.
    def _energy_at_S(S_d: np.ndarray) -> float:
        from ..solvers._casci import casci as _run_casci

        O = C_mo.T @ S_d @ C_mo
        eigvals, eigvecs = np.linalg.eigh(O)
        O_invsqrt = (
            eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
        )
        C_d = C_mo @ O_invsqrt

        h_mo = C_d.T @ h_ao @ C_d
        g_pa = g_ao.transpose(0, 2, 1, 3)
        g_mo_phys = np.einsum("ap,bq,cr,ds,abcd->pqrs", C_d, C_d, C_d, C_d, g_pa)

        cas = _run_casci(
            h_mo,
            g_mo_phys,
            n_act_elec,
            n_act,
            n_core=n_core,
            nuclear_repulsion=mol.nuclear_repulsion(),
            ms2=ms2,
        )
        return cas.e_total

    # FD for each (mu, ν) pair
    W_num = np.zeros((nb, nb))
    for mu in range(nb):
        for nu in range(mu, nb):
            dS = np.zeros((nb, nb))
            dS[mu, nu] = fd_eps
            dS[nu, mu] = fd_eps  # symmetric perturbation
            E_plus = _energy_at_S(S_ref + dS)
            E_minus = _energy_at_S(S_ref - dS)
            W_num[mu, nu] = -(E_plus - E_minus) / (2.0 * fd_eps)
            W_num[nu, mu] = W_num[mu, nu]

    # Factor-of-2 correction: symmetric perturbation doubles the energy
    # response for OFF-DIAGONAL elements (both mu,nu and nu,mu are set).
    # Diagonal elements (mu==nu) are NOT doubled -- only one index is perturbed.
    W = W_num.copy()
    for mu in range(nb):
        for nu in range(mu + 1, nb):
            W[mu, nu] /= 2.0
            W[nu, mu] /= 2.0
    return W


# ---------------------------------------------------------------------------
#  Orbital-rotation / CI-response helpers.  NOT used by the CASSCF gradient
#  itself (a variational energy needs no response); they are the building
#  blocks of the CASPT2 / NEVPT2 relaxed gradients (vibeqc.gradient._caspt2,
#  _nevpt2, _pt2_lagrangian, _pt2_wz_explicit, _ms_caspt2_grad), which
#  solve the Handy-Schaefer z-vector equations (Helgaker Sec. 12.5.24-27,
#  Handy & Schaefer JCP 81, 5031 (1984)) for their non-variational energies.
# ---------------------------------------------------------------------------


def _nonredundant_pairs_local(
    n_core: int, n_act: int, norb: int
) -> list[tuple[int, int]]:
    """Non-redundant orbital-rotation pairs (p, q) with p above q.

    Same convention as :func:`vibeqc.solvers._casscf._nonredundant_pairs`.
    Copied here to avoid a solver->gradient import cycle.
    """
    inact = range(0, n_core)
    act = range(n_core, n_core + n_act)
    virt = range(n_core + n_act, norb)
    pairs: list[tuple[int, int]] = []
    for i in inact:
        for t in act:
            pairs.append((t, i))
        for a in virt:
            pairs.append((a, i))
    for t in act:
        for a in virt:
            pairs.append((a, t))
    return pairs


def _build_kappa_from_z(
    z: np.ndarray, pairs: list[tuple[int, int]], norb: int
) -> np.ndarray:
    """Antisymmetric orbital rotation matrix κ from packed z-vector."""
    K = np.zeros((norb, norb))
    for (p, q), v in zip(pairs, z):
        K[p, q] = v
        K[q, p] = -v
    return K


def _build_full_generalized_fock(
    h1_mo: np.ndarray,
    eri_chem: np.ndarray,
    dm1_active: np.ndarray,
    n_core: int,
    n_act: int,
) -> np.ndarray:
    """Symmetric generalized Fock matrix (1-RDM only).

    F = h + core mean field + active(1-RDM) mean field.
    Same formula as :func:`vibeqc.solvers._mrpt._generalized_fock`.
    Returns a symmetric matrix (F_pq = F_qp).
    """
    F = h1_mo.copy()
    for i in range(n_core):
        F += 2.0 * eri_chem[:, :, i, i] - eri_chem[:, i, i, :]
    aidx = range(n_core, n_core + n_act)
    for ti, t in enumerate(aidx):
        for ui, u in enumerate(aidx):
            d = dm1_active[ti, ui]
            if abs(d) > 1e-14:
                F += d * (eri_chem[:, :, t, u] - 0.5 * eri_chem[:, t, u, :])
    return F


def _build_casscf_fock_and_gradient(
    h1: np.ndarray,
    eri_chem: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    n_core: int,
    n_act: int,
    norb: int,
    pairs: list[tuple[int, int]],
):
    """CASSCF orbital gradient g_pq = 2(F_qp - F_pq) and generalized Fock F.

    This is the SAME construction used by the CASSCF solver's
    ``_orbital_gradient_and_fock`` -- essential for the correct non-zero
    orbital gradient.  The F matrix is:
      * Core rows: F[i,:] = 2.Favg[i,:]  (inactive Fock)
      * Active rows: F[t,:] = dm1.FI + dm2.g  (involves 2-RDM)
    where Favg is the 1-RDM generalized Fock and FI is the inactive Fock.

    Uses the fused C++ kernel (:func:`casscf_orbital_gradient`) when
    available; falls back to a Python einsum path.

    Returns (grad, F, Favg).
    """
    # C++ fast path — same kernel used by the CASSCF solver.
    try:
        from .._vibeqc_core import casscf_orbital_gradient as _casscf_grad

        eri_flat = np.ascontiguousarray(eri_chem.ravel(), dtype=float)
        dm2_flat = np.ascontiguousarray(dm2.ravel(), dtype=float)
        dm1_arr = np.ascontiguousarray(dm1, dtype=float)
        pairs_vec = [(int(p), int(q)) for p, q in pairs]
        result = _casscf_grad(
            np.ascontiguousarray(h1, dtype=float),
            eri_flat,
            dm1_arr,
            dm2_flat,
            n_core,
            n_act,
            norb,
            pairs_vec,
        )
        return np.asarray(result[0]), np.asarray(result[1]), np.asarray(result[2])
    except (ImportError, AttributeError):
        pass

    # Python fallback.
    A = slice(n_core, n_core + n_act)

    # Inactive Fock: FI = h1 + core mean field
    FI = h1.copy()
    for i in range(n_core):
        FI += 2.0 * eri_chem[:, :, i, i] - eri_chem[:, i, i, :]

    # Symmetric generalized Fock (1-RDM only) -- used for core rows
    Favg = _build_full_generalized_fock(h1, eri_chem, dm1, n_core, n_act)

    F = np.zeros((norb, norb))
    if n_core:
        F[:n_core, :] = 2.0 * Favg[:n_core, :]
    if n_act:
        # Active row: F_tq = S_u dm1_tu.FI_uq + S_{uvw} dm2_tuvw.(qu|vw)
        F1 = np.einsum("tu,qu->tq", dm1, FI[:, A], optimize=True)
        F2 = np.einsum("tuvw,quvw->tq", dm2, eri_chem[:, A, A, A], optimize=True)
        F[A, :] = F1 + F2

    grad = np.array([2.0 * (F[q, p] - F[p, q]) for (p, q) in pairs])

    return grad, F, Favg


# ---------------------------------------------------------------------------
#  Energy-FD Hessian and g^R (fixed RDMs, correct CP-MCSCF RHS)
# ---------------------------------------------------------------------------


def _energy_at_rotated_fixed_rdm(
    h1_mo: np.ndarray,
    g_phys_mo: np.ndarray,
    kappa: np.ndarray,
    D_mo: np.ndarray,
    gamma_mo: np.ndarray,
    nuclear_repulsion: float,
) -> float:
    """CASSCF energy at rotated MOs with frozen CI coefficients.

    Rotates only the MO integrals (U = exp(κ)); the RDMs stay in the
    original MO basis.  This is the correct energy for the CP-MCSCF
    energy cross-derivative: the wavefunction parameters (RDMs) are
    held fixed while the Hamiltonian rotates.
    """
    U = expm(kappa)
    h1r = U.T @ h1_mo @ U
    g_phys_r = np.einsum("ap,bq,cr,ds,abcd->pqrs", U, U, U, U, g_phys_mo, optimize=True)
    e1 = np.sum(h1r * D_mo)
    # ``g_phys_mo`` uses the solver's physicist ordering, while the full
    # spin-summed RDM uses the chemist ordering expected by
    # ``energy_from_rdms`` and the analytic derivative contractions.
    eri_chem_r = g_phys_r.transpose(0, 2, 1, 3)
    e2 = 0.5 * np.sum(eri_chem_r * gamma_mo)
    return nuclear_repulsion + e1 + e2


def _orbital_gradient_energy_fd(
    h1_mo: np.ndarray,
    g_phys_mo: np.ndarray,
    D_mo: np.ndarray,
    gamma_mo: np.ndarray,
    nuclear_repulsion: float,
    pairs: list[tuple[int, int]],
    norb: int,
    fd_eps: float = 1e-4,
) -> np.ndarray:
    """Orbital gradient by central FD of energy at rotated MOs.

    Computes g_i = [E(+e e_i) - E(-e e_i)] / (2e) where E(κ) is the
    energy at rotated MOs with *fixed* RDMs in the original basis.

    Returns g of shape (n_pairs,).
    """
    npr = len(pairs)
    grad = np.zeros(npr)
    for i in range(npr):
        e_i = np.zeros(npr)
        e_i[i] = 1.0
        kp = _build_kappa_from_z(fd_eps * e_i, pairs, norb)
        km = _build_kappa_from_z(-fd_eps * e_i, pairs, norb)
        Ep = _energy_at_rotated_fixed_rdm(
            h1_mo, g_phys_mo, kp, D_mo, gamma_mo, nuclear_repulsion
        )
        Em = _energy_at_rotated_fixed_rdm(
            h1_mo, g_phys_mo, km, D_mo, gamma_mo, nuclear_repulsion
        )
        grad[i] = (Ep - Em) / (2.0 * fd_eps)
    return grad


# ---------------------------------------------------------------------------
#  CI coupling -- H_cc and H_oc blocks of the CP-MCSCF Hessian
# ---------------------------------------------------------------------------


def _build_ci_hamiltonian(
    h1e_cas: np.ndarray,
    h2e_cas: np.ndarray,
    determinants: list,
    n_core: int,
    n_act: int,
) -> np.ndarray:
    """Build the CI Hamiltonian matrix H_det in the unrestricted determinant
    basis with frozen-core dressing.

    Uses the CASCI solver's ``_frozen_core_dressing`` to construct the
    active-space one-electron integrals and ``build_hamiltonian_matrix_unrestricted``
    from the Slater-Condon module for the full CI matrix.

    Returns H_det of shape (n_det, n_det).
    """
    from ..solvers._casci import _frozen_core_dressing
    from ..solvers._slater_condon import build_hamiltonian_matrix_unrestricted

    active = slice(n_core, n_core + n_act)
    e_core, h1e_active = _frozen_core_dressing(h1e_cas, h2e_cas, n_core, active)
    h2e_active = np.ascontiguousarray(h2e_cas[active, active, active, active])
    return build_hamiltonian_matrix_unrestricted(determinants, h1e_active, h2e_active)


def _build_full_gamma_mo(
    rdm1_active: np.ndarray,
    rdm2_active: np.ndarray,
    n_core: int,
    n_act: int,
    nmo: int,
) -> np.ndarray:
    """Build the full MO 2-RDM from core and active contributions.

    Core-core: Γ_ijkl = 4d_ijd_kl - 2d_ild_jk (closed-shell)
    Core-active: Γ_ijtu = 2d_ij g_tu, Γ_ituj = -d_ij g_tu
    Active-active: Γ_tuvw from CASCI 2-RDM.
    """
    gamma = np.zeros((nmo, nmo, nmo, nmo))
    for i in range(n_core):
        for j in range(n_core):
            for k in range(n_core):
                for l in range(n_core):
                    gamma[i, j, k, l] = 4.0 * (i == j) * (k == l) - 2.0 * (i == l) * (
                        j == k
                    )
    # Core-active cross terms: rdm2[p,q,r,s] = <a+_p a+_r a_s a_q>
    # For core i and active t,u: factorize <core|a+_i a_j|core> x <active|a+_t a_u|active>
    #   rdm2[i,i,t,u] = 2 * D_active[t,u]    (core,core,act,act)
    #   rdm2[t,u,i,i] = 2 * D_active[t,u]    (act,act,core,core)
    #   rdm2[i,t,u,i] = -D_active[t,u]       (core,act,act,core, exchange)
    #   rdm2[t,i,i,u] = -D_active[t,u]       (act,core,core,act, exchange)
    for i in range(n_core):
        for t in range(n_act):
            it = n_core + t
            for u in range(n_act):
                iu = n_core + u
                val = rdm1_active[t, u]
                if abs(val) > 1e-15:
                    gamma[i, i, it, iu] += 2.0 * val
                    gamma[it, iu, i, i] += 2.0 * val
                    gamma[i, it, iu, i] -= val
                    gamma[it, i, i, iu] -= val
    A = slice(n_core, n_core + n_act)
    gamma[A, A, A, A] = rdm2_active
    return gamma


def _build_ci_orbital_coupling_energy_fd(
    h1_mo: np.ndarray,
    g_phys_mo: np.ndarray,
    h1e_cas: np.ndarray,
    h2e_cas: np.ndarray,
    D_mo: np.ndarray,
    gamma_mo: np.ndarray,
    nuclear_repulsion: float,
    ci_coeffs: np.ndarray,
    determinants: list,
    n_core: int,
    n_act: int,
    pairs: list[tuple[int, int]],
    norb: int,
    fd_eps_ci: float = 5e-3,
    fd_eps_grad: float = 1e-4,
) -> np.ndarray:
    """Orbital-CI coupling H_oc by FD of the orbital gradient w.r.t.
    CI state rotations.

    Works in the eigenbasis of the CI Hamiltonian H_det.  For each excited
    CI state |k> (k > 0), the orbital gradient is computed at the perturbed
    CI vector c(th) = cos(th)|0> + sin(th)|k>.  The orbital gradient is then
    FD'd with respect to th.

    Returns H_oc of shape (n_pairs, n_det - 1).
    """
    from ..solvers._rdm import make_rdm12

    npr = len(pairs)
    n_det = len(determinants)

    # Build CI Hamiltonian and diagonalize
    H_det = _build_ci_hamiltonian(h1e_cas, h2e_cas, determinants, n_core, n_act)
    _, U_det = np.linalg.eigh(H_det)

    # Reference RDMs (ground state)
    rdm1_ref, rdm2_ref = make_rdm12(ci_coeffs, determinants, n_act)

    H_oc = np.zeros((npr, n_det - 1))
    for k in range(1, n_det):
        ek = U_det[:, k].copy()
        c_plus = np.cos(fd_eps_ci) * ci_coeffs + np.sin(fd_eps_ci) * ek
        c_plus = c_plus / np.linalg.norm(c_plus)
        c_minus = np.cos(fd_eps_ci) * ci_coeffs - np.sin(fd_eps_ci) * ek
        c_minus = c_minus / np.linalg.norm(c_minus)

        for sign, cv in [(+1, c_plus), (-1, c_minus)]:
            rdm1_p, rdm2_p = make_rdm12(cv, determinants, n_act)
            # Build full D_mo and gamma_mo with core contributions
            D_full = np.zeros_like(D_mo)
            D_full[:n_core, :n_core] = np.diag(np.full(n_core, 2.0))
            A_act = slice(n_core, n_core + n_act)
            D_full[A_act, A_act] = rdm1_p

            gamma_full = _build_full_gamma_mo(rdm1_p, rdm2_p, n_core, n_act, norb)

            g_p = _orbital_gradient_energy_fd(
                h1_mo,
                g_phys_mo,
                D_full,
                gamma_full,
                nuclear_repulsion,
                pairs,
                norb,
                fd_eps_grad,
            )
            if sign == 1:
                g_plus = g_p
            else:
                g_minus = g_p

        H_oc[:, k - 1] = (g_plus - g_minus) / (2.0 * fd_eps_ci)

    return H_oc


def _rotate_integrals(h1: np.ndarray, g_phys: np.ndarray, U: np.ndarray):
    """Rotate MO integrals by unitary U (columns = new orbitals)."""
    h1_new = U.T @ h1 @ U
    g_new = np.einsum("ap,bq,cr,ds,abcd->pqrs", U, U, U, U, g_phys, optimize=True)
    return h1_new, g_new


def _compute_orbital_gradient_at_rotated(
    h1_mo: np.ndarray,
    g_phys_mo: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    D_mo_full: np.ndarray,
    gamma_mo_full: np.ndarray,
    n_core: int,
    n_act: int,
    norb: int,
    pairs: list[tuple[int, int]],
    kappa: np.ndarray,
) -> np.ndarray:
    """Orbital gradient at geometry where MOs are rotated by exp(κ).

    Uses FROZEN RDMs transformed to the rotated basis -- no CASCI re-solve.
    Uses the CASSCF solver's Fock construction (involves 2-RDM).
    """
    U = expm(kappa)
    # Rotate MO integrals
    h1r, gr = _rotate_integrals(h1_mo, g_phys_mo, U)
    eri_chem = gr.transpose(0, 2, 1, 3)  # physicist's -> chemist's
    # Also transform the density matrices to the rotated basis
    D_mo_rot = U.T @ D_mo_full @ U
    gamma_rot = np.einsum(
        "pa,qb,rc,sd,abcd->pqrs", U, U, U, U, gamma_mo_full, optimize=True
    )
    # Active-space RDMs in the rotated basis
    A = slice(n_core, n_core + n_act)
    dm1_rot = D_mo_rot[A, A]
    dm2_rot = gamma_rot[A, A, A, A]
    # Use CASSCF solver's Fock construction (involves 2-RDM for active rows)
    grad, _F, _Favg = _build_casscf_fock_and_gradient(
        h1r, eri_chem, dm1_rot, dm2_rot, n_core, n_act, norb, pairs
    )
    return grad


def _build_orbital_hessian_fd(
    h1_mo: np.ndarray,
    g_phys_mo: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    D_mo_full: np.ndarray,
    gamma_mo_full: np.ndarray,
    n_core: int,
    n_act: int,
    norb: int,
    pairs: list[tuple[int, int]],
    fd_eps: float = 1e-4,
) -> np.ndarray:
    """Build the orbital-orbital Hessian H via FD-HVP.

    H_{ij} = (g(κ=+e e_j) - g(κ=-e e_j))_i / (2e)

    where e_j is the j-th unit vector in the non-redundant pair space
    and g(κ) is the orbital gradient at rotated MO integrals (frozen RDMs).

    Returns H of shape (n_pairs, n_pairs).
    """
    npr = len(pairs)
    H = np.zeros((npr, npr))
    for j in range(npr):
        e_j = np.zeros(npr)
        e_j[j] = 1.0
        kp = _build_kappa_from_z(fd_eps * e_j, pairs, norb)
        km = _build_kappa_from_z(-fd_eps * e_j, pairs, norb)
        gp = _compute_orbital_gradient_at_rotated(
            h1_mo,
            g_phys_mo,
            dm1,
            dm2,
            D_mo_full,
            gamma_mo_full,
            n_core,
            n_act,
            norb,
            pairs,
            kp,
        )
        gm = _compute_orbital_gradient_at_rotated(
            h1_mo,
            g_phys_mo,
            dm1,
            dm2,
            D_mo_full,
            gamma_mo_full,
            n_core,
            n_act,
            norb,
            pairs,
            km,
        )
        H[:, j] = (gp - gm) / (2.0 * fd_eps)
    # Symmetrize to clean up FD noise
    H = 0.5 * (H + H.T)
    return H


def _compute_perturbed_orbital_gradient_fd(
    mol,
    basis,
    C_mo: np.ndarray,
    h1_mo: np.ndarray,
    g_phys_mo: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    D_mo_full: np.ndarray,
    gamma_mo_full: np.ndarray,
    n_core: int,
    n_act: int,
    norb: int,
    pairs: list[tuple[int, int]],
    fd_eps: float = 1e-3,
    *,
    dm1_correction: np.ndarray | None = None,
    dm2_correction: np.ndarray | None = None,
    dm_correction_callable: callable | None = None,
) -> np.ndarray:
    """Compute the perturbed orbital gradient g^R = dg/dR by numerical FD.

    For each atom a and coordinate c, the orbital gradient is computed at
    displaced geometries R_a ± e.  The MO coefficients are re-orthonormalized
    at each displaced geometry (C_d = C . (C^T S_d C)^{-1/2}), the MO integrals
    are transformed to the new orthonormal basis, and the RDMs are also
    transformed to that basis (frozen CI coefficients, no CASCI re-solve).

    When ``dm1_correction`` / ``dm2_correction`` are provided, they are added
    to the active-space RDMs before Fock construction.  This allows computing
    a PT2-corrected g^R without recomputing the full PT2 energy at each
    displaced geometry.

    Returns g^R of shape (n_pairs, 3 * n_atoms).
    """
    n_atoms = len(mol.atoms)
    npr = len(pairs)
    atoms_list = list(mol.atoms)

    g_R = np.zeros((npr, 3 * n_atoms))

    for a in range(n_atoms):
        for c in range(3):
            xyz_p = np.array([at.xyz for at in atoms_list], dtype=float)
            xyz_m = np.array([at.xyz for at in atoms_list], dtype=float)
            xyz_p[a, c] += fd_eps
            xyz_m[a, c] -= fd_eps

            mol_p = Molecule(
                [Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz_p)]
            )
            mol_m = Molecule(
                [Atom(int(at.Z), list(xyz)) for at, xyz in zip(atoms_list, xyz_m)]
            )
            basis_p = BasisSet(mol_p, basis.name)
            basis_m = BasisSet(mol_m, basis.name)

            S_p = np.asarray(_compute_overlap(basis_p))
            S_m = np.asarray(_compute_overlap(basis_m))

            # Re-orthonormalize MOs at displaced geometry.
            # C_d = C @ O^{-1/2} where O = C^T S_d C
            # The transformation T = O^{-1/2} maps reference->displaced orthonormal basis.
            # RDMs transform as D' = T^{-1} D T^{-T} = O^{1/2} D O^{1/2}.
            def _prepare(S_d):
                O = C_mo.T @ S_d @ C_mo
                eigvals, eigvecs = np.linalg.eigh(O)
                O_invsqrt = (
                    eigvecs
                    @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12)))
                    @ eigvecs.T
                )
                O_sqrt = (
                    eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T
                )
                C_d = C_mo @ O_invsqrt
                D_d = O_sqrt @ D_mo_full @ O_sqrt
                gamma_d = np.einsum(
                    "pa,qb,rc,sd,abcd->pqrs",
                    O_sqrt,
                    O_sqrt,
                    O_sqrt,
                    O_sqrt,
                    gamma_mo_full,
                    optimize=True,
                )
                return C_d, D_d, gamma_d, O_sqrt

            Cd_p, Dd_p, gammad_p, O_sqrt_p = _prepare(S_p)
            Cd_m, Dd_m, gammad_m, O_sqrt_m = _prepare(S_m)

            h_p = np.asarray(_compute_kinetic(basis_p)) + np.asarray(
                _compute_nuclear(basis_p, mol_p)
            )
            h_m = np.asarray(_compute_kinetic(basis_m)) + np.asarray(
                _compute_nuclear(basis_m, mol_m)
            )
            g_ao_p = np.asarray(_compute_eri(basis_p))  # chemist's
            g_ao_m = np.asarray(_compute_eri(basis_m))

            h_mo_p = Cd_p.T @ h_p @ Cd_p
            h_mo_m = Cd_m.T @ h_m @ Cd_m
            g_pa_p = g_ao_p.transpose(0, 2, 1, 3)  # chemist's -> physicist's
            g_pa_m = g_ao_m.transpose(0, 2, 1, 3)
            g_mo_p = np.einsum(
                "ap,bq,cr,ds,abcd->pqrs", Cd_p, Cd_p, Cd_p, Cd_p, g_pa_p, optimize=True
            )
            g_mo_m = np.einsum(
                "ap,bq,cr,ds,abcd->pqrs", Cd_m, Cd_m, Cd_m, Cd_m, g_pa_m, optimize=True
            )

            eri_c_p = g_mo_p.transpose(0, 2, 1, 3)  # physicist's -> chemist's
            eri_c_m = g_mo_m.transpose(0, 2, 1, 3)

            dm1_p = Dd_p[n_core : n_core + n_act, n_core : n_core + n_act]
            dm1_m = Dd_m[n_core : n_core + n_act, n_core : n_core + n_act]
            # Also transform dm2 to the displaced basis
            A_slc = slice(n_core, n_core + n_act)
            dm2_p = gammad_p[A_slc, A_slc, A_slc, A_slc]
            dm2_m = gammad_m[A_slc, A_slc, A_slc, A_slc]

            # Apply PT2 effective density corrections if provided.
            # Two modes:
            #   1. Constant corrections (dm1_correction, dm2_correction):
            #      assumed invariant in MO basis — fast, 99.99% accurate.
            #   2. Per-geometry callable (dm_correction_callable):
            #      recomputes corrections at each displaced geometry
            #      via the PT2 solver — captures dDeltaD/dR, dDeltaGamma/dR.
            _dm1c = dm1_correction
            _dm2c = dm2_correction
            if dm_correction_callable is not None:
                _dm1c, _dm2c = dm_correction_callable(h_mo_p, eri_c_p, n_core, n_act)
                _dm1c_m, _dm2c_m = dm_correction_callable(
                    h_mo_m, eri_c_m, n_core, n_act
                )
                if _dm1c is not None:
                    dm1_p = dm1_p + _dm1c
                if _dm2c is not None:
                    dm2_p = dm2_p + _dm2c
                if _dm1c_m is not None:
                    dm1_m = dm1_m + _dm1c_m
                if _dm2c_m is not None:
                    dm2_m = dm2_m + _dm2c_m
            else:
                if _dm1c is not None:
                    dm1_p = dm1_p + _dm1c
                    dm1_m = dm1_m + _dm1c
                if _dm2c is not None:
                    dm2_p = dm2_p + _dm2c
                    dm2_m = dm2_m + _dm2c

            g_p, _Fp, _Favgp = _build_casscf_fock_and_gradient(
                h_mo_p, eri_c_p, dm1_p, dm2_p, n_core, n_act, norb, pairs
            )
            g_m, _Fm, _Favgm = _build_casscf_fock_and_gradient(
                h_mo_m, eri_c_m, dm1_m, dm2_m, n_core, n_act, norb, pairs
            )

            col = a * 3 + c
            g_R[:, col] = (g_p - g_m) / (2.0 * fd_eps)

    return g_R


def _compute_one_index_transformed_dm(
    kappa: np.ndarray, D_mo: np.ndarray
) -> np.ndarray:
    """One-index transformed 1-RDM: D^z = [κ, D] = κ.D - D.κ."""
    return kappa @ D_mo - D_mo @ kappa


def _compute_one_index_transformed_gamma(
    kappa: np.ndarray, gamma_mo: np.ndarray
) -> np.ndarray:
    """One-index transformed 2-RDM: Γ^z_{pqrs} = S_t (κ_{pt}Γ_{tqrs} + ...).

    The 4-index one-index transform under antisymmetric κ:
      Γ^z_{pqrs} = κ_{pt} Γ_{tqrs} + κ_{qt} Γ_{ptrs} + κ_{rt} Γ_{pqts} + κ_{st} Γ_{pqrt}
    (Einstein summation over t).
    """
    nmo = gamma_mo.shape[0]
    gamma_z = np.zeros_like(gamma_mo)
    for p in range(nmo):
        for q in range(nmo):
            for r in range(nmo):
                for s in range(nmo):
                    val = 0.0
                    for t in range(nmo):
                        kpt = kappa[p, t]
                        kqt = kappa[q, t]
                        krt = kappa[r, t]
                        kst = kappa[s, t]
                        if abs(kpt) > 1e-15:
                            val += kpt * gamma_mo[t, q, r, s]
                        if abs(kqt) > 1e-15:
                            val += kqt * gamma_mo[p, t, r, s]
                        if abs(krt) > 1e-15:
                            val += krt * gamma_mo[p, q, t, s]
                        if abs(kst) > 1e-15:
                            val += kst * gamma_mo[p, q, r, t]
                    gamma_z[p, q, r, s] = val
    return gamma_z


def _compute_wz_mo(F: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Energy-weighted density correction from orbital relaxation.

    W^z_MO = F.κ - κ.F  (commutator of generalized Fock with κ).

    Only the symmetric part of W^z contributes to tr(W^z.dS/dR).
    We return the full matrix (not symmetrized) because the contraction
    with dS (symmetric) implicitly picks out the symmetric part.
    """
    return F @ kappa - kappa @ F


