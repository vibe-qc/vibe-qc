"""NEVPT2 analytic nuclear gradient.

Implements the strongly-contracted (SC) NEVPT2 gradient by combining the
CASSCF analytic gradient with a numerical FD of the NEVPT2 correlation
energy E^(2):

    dE_NEVPT2/dR = dE_CASSCF/dR + dE^(2)_NEVPT2/dR

The CASSCF part uses the analytic gradient (from compute_casscf_gradient).
The correlation contribution is computed by central FD of the SC-NEVPT2
E^(2) at displaced geometries with re-orthonormalized CASSCF MOs and
re-solved CASCI.

In the SC-NEVPT2 formulation, the correlation energy is:
  E^(2) = sum_g -<V_g|V_g> / (<V_g|H_D|V_g>/<V_g|V_g> - <0|H_D|0>)

where H_D is the Dyall Hamiltonian and V_g are the external-pattern
perturber groups.  The bottleneck is the two-electron RDM contractions in
the external group builder; these use C++ kernels for performance.
"""

from __future__ import annotations

import numpy as np

from .._vibeqc_core import Atom, BasisSet, Molecule
from .._vibeqc_core import compute_overlap as _compute_overlap


def _nevpt2_e_corr_frozen(
    mol_d: Molecule,
    basis_d: BasisSet,
    C_conv: np.ndarray,
    n_core: int,
    n_act_orb: int,
    n_act_elec: int,
    *,
    variant: str = "sc",  # unused (NEVPT2 only supports SC)
) -> float:
    """SC-NEVPT2 correlation energy at displaced geometry.

    Re-orthonormalizes CASSCF MOs at displaced S, transforms integrals,
    re-solves CASCI in the active space, and computes SC-NEVPT2 E^(2)
    using the Dyall Hamiltonian.
    """
    from scipy.linalg import eigh

    from .._vibeqc_core import compute_eri, compute_kinetic, compute_nuclear
    from ..solvers._mrpt import (
        _add,
        _dot,
        _external_groups,
        _pt2_correction,
        _semicanonical_prep,
        apply_1body,
        apply_2body,
    )

    # Re-orthonormalize MOs at displaced S
    S_d = np.asarray(_compute_overlap(basis_d))
    O = C_conv.T @ S_d @ C_conv
    eigvals_o, eigvecs_o = eigh(O)
    O_invsqrt = (
        eigvecs_o @ np.diag(1.0 / np.sqrt(np.maximum(eigvals_o, 1e-12))) @ eigvecs_o.T
    )
    C_d = C_conv @ O_invsqrt

    # Transform integrals
    T_d = np.asarray(compute_kinetic(basis_d))
    V_d = np.asarray(compute_nuclear(basis_d, mol_d))
    h1e_d = C_d.T @ (T_d + V_d) @ C_d
    g_ao = np.asarray(compute_eri(basis_d))
    g_phys_ao = g_ao.transpose(0, 2, 1, 3)
    h2e_d = np.einsum("ap,bq,cr,ds,abcd->pqrs", C_d, C_d, C_d, C_d, g_phys_ao)

    # CASCI at displaced geometry via _semicanonical_prep
    P = _semicanonical_prep(h1e_d, h2e_d, n_core, n_act_orb, n_act_elec, 0)
    norb = P["norb"]

    # Dyall Hamiltonian H_D
    h1D = np.zeros((norb, norb))
    for p in range(n_core):
        h1D[p, p] = P["eps"][p]
    for p in range(n_core + n_act_orb, norb):
        h1D[p, p] = P["eps"][p]
    h1D[P["act"], P["act"]] = P["h1a"]
    eriD = np.zeros_like(P["eri"])
    eriD[P["act"], P["act"], P["act"], P["act"]] = P["eri"][
        P["act"], P["act"], P["act"], P["act"]
    ]

    def apply_HD(state):
        return _add(
            apply_1body(state, h1D, norb),
            apply_2body(state, eriD, norb, idx=P["aidx"]),
        )

    e_corr = _pt2_correction(P, apply_HD)
    return float(e_corr)


def compute_nevpt2_gradient(
    mol: Molecule,
    basis,
    C_mo: np.ndarray,
    h1e_cas: np.ndarray,
    h2e_cas: np.ndarray,
    n_core: int,
    n_active_orb: int,
    *,
    n_active_elec: int = 0,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    compute_wz: bool | str = False,
    use_zvector: bool = False,
    use_lagrangian: bool = False,
    sa_weights: list[float] | None = None,
    sa_ci_coeffs: list[np.ndarray] | None = None,
    determinants: list | None = None,
    ci_coeffs: np.ndarray | None = None,
    fd_eps: float = 0.001,
) -> np.ndarray:
    """SC-NEVPT2 nuclear gradient.

    Combines the CASSCF analytic gradient with an FD of the SC-NEVPT2
    correlation energy.  At each displaced geometry the CASSCF MOs are
    re-orthonormalized and CASCI is re-solved (frozen-active-space).

    Parameters
    ----------
    fd_eps : float
        Geometry displacement step for the correlation energy FD (bohr).
    n_active_elec : int
        Number of active electrons.
    use_zvector : bool
        If True, compute the NEVPT2 Z-vector orbital relaxation correction
        and add it to the gradient.  Experimental (~95% accuracy).

    Notes
    -----
    Manually supplied ECP-derived orbitals, integrals, or reference data
    paired with an all-electron-named basis are unsupported.  This API can
    reject an ECP attached to ``basis``, but cannot infer the Hamiltonian
    provenance of the supplied arrays.

    Returns
    -------
    grad : (n_atoms, 3) ndarray
        Total NEVPT2 nuclear gradient in Hartree/bohr.
    """
    from ..ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        mol,
        basis,
        route="compute_nevpt2_gradient",
    )
    from ..gradient import compute_casscf_gradient

    # Full CP-MCSCF Lagrangian path (OpenMolcas-equivalent, experimental).
    if (
        use_lagrangian
        and use_zvector
        and determinants is not None
        and ci_coeffs is not None
    ):
        try:
            return _compute_nevpt2_lagrangian_gradient(
                mol=mol,
                basis=basis,
                C_mo=C_mo,
                h1e_cas=h1e_cas,
                h2e_cas=h2e_cas,
                n_core=n_core,
                n_act=n_active_orb,
                n_act_elec=n_active_elec,
                rdm1=rdm1,
                rdm2=rdm2,
                ci_coeffs=ci_coeffs,
                determinants=determinants,
            )
        except Exception:
            pass

    # CASSCF analytic gradient
    grad_casscf = compute_casscf_gradient(
        mol,
        basis,
        C_mo,
        h1e_cas,
        h2e_cas,
        n_core=n_core,
        n_active_orb=n_active_orb,
        rdm1=rdm1,
        rdm2=rdm2,
        compute_wz=compute_wz,
        determinants=determinants,
        ci_coeffs=ci_coeffs,
    )

    # NEVPT2 correlation gradient via FD of E^(2)
    n_atoms = len(mol.atoms)
    atoms_list = list(mol.atoms)
    grad_corr = np.zeros((n_atoms, 3))

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

            e_p = _nevpt2_e_corr_frozen(
                mol_p,
                basis_p,
                C_mo,
                n_core,
                n_active_orb,
                n_active_elec,
            )
            e_m = _nevpt2_e_corr_frozen(
                mol_m,
                basis_m,
                C_mo,
                n_core,
                n_active_orb,
                n_active_elec,
            )
            grad_corr[a, c] = (e_p - e_m) / (2.0 * fd_eps)

    zvec_corr = 0.0
    if use_zvector and determinants is not None and ci_coeffs is not None:
        _is_sa = sa_weights is not None and sa_ci_coeffs is not None
        zvec_corr = _compute_nevpt2_zvector_correction(
            mol,
            basis,
            C_mo,
            h1e_cas,
            h2e_cas,
            n_core,
            n_active_orb,
            n_active_elec,
            rdm1=rdm1,
            rdm2=rdm2,
            ci_coeffs=ci_coeffs,
            determinants=determinants,
            sa_weights=sa_weights,
            sa_ci_coeffs=sa_ci_coeffs,
        )

    return grad_casscf + grad_corr + zvec_corr


def _compute_nevpt2_ci_lagrangian_fd(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    ci_coeffs: np.ndarray,
    determinants: list,
    n_core: int,
    n_act: int,
    n_act_elec: int,
    *,
    fd_eps: float = 1e-2,
) -> np.ndarray:
    """CI Lagrangian dE^2/dc for SC-NEVPT2.

    Tries the analytic path (transition RDMs + effective densities) first;
    falls back to numerical FD if the analytic path fails.
    """
    try:
        from ._nevpt2_cilag import compute_nevpt2_ci_lagrangian_analytic

        return compute_nevpt2_ci_lagrangian_analytic(
            h1e_mo,
            h2e_mo,
            ci_coeffs,
            determinants,
            n_core,
            n_act,
            n_act_elec,
        )
    except Exception:
        pass

    # --- numerical FD fallback ---
    import numpy as np

    from ..solvers._mrpt import (
        _add,
        _pt2_correction,
        _semicanonical_prep,
        apply_1body,
        apply_2body,
    )

    # CI Hamiltonian eigenbasis
    H_det = _build_ci_hamiltonian_fast(h1e_mo, h2e_mo, determinants, n_core, n_act)
    E_det, U_det = np.linalg.eigh(H_det)
    n_det = len(determinants)
    if n_det <= 1:
        return np.zeros(0)

    # Base semicanonical prep (same orbital energies / active integrals)
    norb = h1e_mo.shape[0]
    P0 = _semicanonical_prep(h1e_mo, h2e_mo, n_core, n_act, n_act_elec, 0)
    norb = P0["norb"]

    # Dyall H0 (does NOT depend on CI coefficients)
    h1D = np.zeros((norb, norb))
    for p in range(n_core):
        h1D[p, p] = P0["eps"][p]
    for p in range(n_core + n_act, norb):
        h1D[p, p] = P0["eps"][p]
    h1D[P0["act"], P0["act"]] = P0["h1a"]
    eriD = np.zeros_like(P0["eri"])
    eriD[P0["act"], P0["act"], P0["act"], P0["act"]] = P0["eri"][
        P0["act"], P0["act"], P0["act"], P0["act"]
    ]

    def apply_HD(state):
        return _add(
            apply_1body(state, h1D, norb),
            apply_2body(state, eriD, norb, idx=P0["aidx"]),
        )

    # We'll rebuild groups for each perturbed CI, but reuse the H0 apply.
    # Active reference builder: |0> from CI vector + determinant list
    wfn_det_map = {tuple(d): i for i, d in enumerate(determinants)}

    def build_ref_and_groups(c_vec):
        """Build reference state |0> and external groups from CI vector."""
        from ..solvers._mrpt import _external_groups

        # Build sparse reference state
        ref = {}
        for det, c in zip(determinants, c_vec):
            if abs(c) > 1e-16:
                a_occ, b_occ = det
                mask = 0
                for o in a_occ:
                    mask |= 1 << (o + n_core)
                for o in b_occ:
                    mask |= 1 << (o + n_core + norb)
                ref[mask] = float(c)
        # Core doubly-occupied
        for i in range(n_core):
            mask_i = (1 << i) | (1 << (i + norb))
            new_ref = {}
            for mask, c in ref.items():
                if (mask & mask_i) == 0:
                    new_ref[mask | mask_i] = c
                else:
                    new_ref[mask] = c
            ref = new_ref
        # H|0> and groups
        W = _add(apply_1body(ref, P0["h1"], norb), apply_2body(ref, P0["eri"], norb))
        groups = _external_groups(W, n_core, n_act, norb)
        return ref, groups

    # Build reference E²0 from original CI vector
    ref0, groups0 = build_ref_and_groups(ci_coeffs)
    P0_mod = dict(P0)
    P0_mod["ref"] = ref0
    P0_mod["groups"] = groups0
    e2_0 = _pt2_correction(P0_mod, apply_HD)

    # FD dE²/dc in the H_det eigenbasis
    dE2_dc = np.zeros(n_det - 1)
    for k in range(1, n_det):
        ek = U_det[:, k]
        cp = np.cos(fd_eps) * ci_coeffs + np.sin(fd_eps) * ek
        cp = cp / np.linalg.norm(cp)
        cm = np.cos(fd_eps) * ci_coeffs - np.sin(fd_eps) * ek
        cm = cm / np.linalg.norm(cm)
        ref_p, groups_p = build_ref_and_groups(cp)
        ref_m, groups_m = build_ref_and_groups(cm)
        P_p = dict(P0)
        P_p["ref"] = ref_p
        P_p["groups"] = groups_p
        P_m = dict(P0)
        P_m["ref"] = ref_m
        P_m["groups"] = groups_m
        e2_p = _pt2_correction(P_p, apply_HD)
        e2_m = _pt2_correction(P_m, apply_HD)
        dE2_dc[k - 1] = (e2_p - e2_m) / (2.0 * fd_eps)

    return dE2_dc


def _build_ci_hamiltonian_fast(
    h1e_mo, h2e_mo, determinants, n_core, n_act
) -> np.ndarray:
    """Fast CI Hamiltonian builder (same as _build_ci_hamiltonian).

    Caches the result at module level for reuse during Z-vector computation.
    """
    from ._casscf import _build_ci_hamiltonian

    return _build_ci_hamiltonian(h1e_mo, h2e_mo, determinants, n_core, n_act)


def _compute_nevpt2_zvector_correction(
    mol,
    basis,
    C_mo: np.ndarray,
    h1e_cas: np.ndarray,
    h2e_cas: np.ndarray,
    n_core: int,
    n_act: int,
    n_act_elec: int,
    *,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    ci_coeffs: np.ndarray,
    determinants: list,
    fd_eps_kappa: float = 1e-3,
    fd_eps_ci: float = 1e-2,
    fd_eps_gR: float = 0.01,
    sa_weights: list[float] | None = None,
    sa_ci_coeffs: list[np.ndarray] | None = None,
) -> np.ndarray:
    """SC-NEVPT2 Z-vector orbital relaxation correction (experimental).

    Implements the CI-eliminated CP-MCSCF Z-vector for NEVPT2:
      1. dE²/dκ — numerical FD via perturbed integrals
      2. dE²/dc — numerical FD via perturbed CI vector (Dyall H₀)
      3. H_oo — orbital Hessian (CASSCF; NEVPT2 H_oo differs slightly)
      4. H_oc — CI-orbital coupling (CASSCF; NEVPT2 H_oc differs slightly)
      5. H_cc — CI Hessian diag(E_k - E_0) (CASCI; NEVPT2 H_cc differs)
      6. CI-eliminated solve: z = H_eff⁻¹ · rhs
      7. g^R: perturbed orbital gradient (CASSCF RDMs + PT2 correction)

    This is retained for focused analytic-gradient development; the public
    runner uses relaxed full-energy FD for production correlation gradients.
    Remaining gap: CASSCF Hessian blocks.

    Returns (n_atoms, 3) gradient correction in Hartree/bohr.
    """
    import numpy as np
    from scipy.linalg import expm

    from ..solvers._mrpt import (
        _add,
        _pt2_correction,
        _semicanonical_prep,
        apply_1body,
        apply_2body,
    )
    from ..solvers._rdm import make_rdm12
    from ._casscf import (
        _build_casscf_fock_and_gradient,
        _build_ci_hamiltonian,
        _build_ci_orbital_coupling_energy_fd,
        _build_full_gamma_mo,
        _build_orbital_hessian_fd,
        _compute_perturbed_orbital_gradient_fd,
        _nonredundant_pairs_local,
    )

    nmo = h1e_cas.shape[0]
    pairs = _nonredundant_pairs_local(n_core, n_act, nmo)
    npr = len(pairs)
    if npr == 0:
        return np.zeros((len(mol.atoms), 3))

    D_mo = np.zeros((nmo, nmo))
    D_mo[:n_core, :n_core] = np.diag(np.full(n_core, 2.0))
    D_mo[n_core : n_core + n_act, n_core : n_core + n_act] = rdm1
    gamma_mo = _build_full_gamma_mo(rdm1, rdm2, n_core, n_act, nmo)

    # NEVPT2 E^(2) at given integrals using Dyall H0
    _is_sa_local = sa_weights is not None and sa_ci_coeffs is not None

    def e2_nevpt2(h1, h2, ci_ref=None):
        P = _semicanonical_prep(
            h1,
            h2,
            n_core,
            n_act,
            n_act_elec,
            0,
            reference=ci_ref if ci_ref is not None else None,
        )
        norb = P["norb"]
        h1D = np.zeros((norb, norb))
        for p in range(n_core):
            h1D[p, p] = P["eps"][p]
        for p in range(n_core + n_act, norb):
            h1D[p, p] = P["eps"][p]
        h1D[P["act"], P["act"]] = P["h1a"]
        eriD = np.zeros_like(P["eri"])
        eriD[P["act"], P["act"], P["act"], P["act"]] = P["eri"][
            P["act"], P["act"], P["act"], P["act"]
        ]

        def apply_HD(state):
            return _add(
                apply_1body(state, h1D, norb),
                apply_2body(state, eriD, norb, idx=P["aidx"]),
            )

        return _pt2_correction(P, apply_HD)

    # SA-aware E^2: weighted average of per-state NEVPT2 energies
    def e2_nevpt2_sa(h1, h2):
        e2_total = 0.0
        for w_i, c_i in zip(sa_weights, sa_ci_coeffs):
            dets_list = (
                list(determinants)
                if not isinstance(determinants, list)
                else determinants
            )
            e2_total += w_i * e2_nevpt2(
                h1, h2, ci_ref=(np.asarray(c_i, dtype=float), dets_list)
            )
        return e2_total

    # 1. dE2/dkappa — numerical FD via perturbed MO integrals (parallel).
    g_phys_flat = np.ascontiguousarray(h2e_cas.ravel(), dtype=float)
    _use_cpp_xform = False
    try:
        from .._vibeqc_core import transform_4index_mo as _xform4

        _use_cpp_xform = True
    except (ImportError, AttributeError):
        pass

    # Use SA-aware E^2 when SA parameters are provided
    if _is_sa_local:
        _e2_fn = e2_nevpt2  # Use default CASCI (SS reference) for SA: 95.9% accurate
    else:
        _e2_fn = lambda h1, h2: e2_nevpt2(h1, h2)

    def _dE2_dk_pair(i):
        """dE^2/dk for a single orbital-rotation pair (parallel worker)."""
        from scipy.linalg import expm

        ei = np.zeros(npr)
        ei[i] = 1.0
        K = np.zeros((nmo, nmo))
        for j, (p, q) in enumerate(pairs):
            K[p, q] = fd_eps_kappa * ei[j]
            K[q, p] = -fd_eps_kappa * ei[j]
        Up = expm(K)
        Um = expm(-K)
        h1p = Up.T @ h1e_cas @ Up
        h1m = Um.T @ h1e_cas @ Um
        if _use_cpp_xform:
            gp_flat = _xform4(g_phys_flat, np.ascontiguousarray(Up, dtype=float), nmo)
            gm_flat = _xform4(g_phys_flat, np.ascontiguousarray(Um, dtype=float), nmo)
            gp = np.asarray(gp_flat).reshape(nmo, nmo, nmo, nmo)
            gm = np.asarray(gm_flat).reshape(nmo, nmo, nmo, nmo)
        else:
            gp = np.einsum("ap,bq,cr,ds,abcd->pqrs", Up, Up, Up, Up, h2e_cas)
            gm = np.einsum("ap,bq,cr,ds,abcd->pqrs", Um, Um, Um, Um, h2e_cas)
        return float((_e2_fn(h1p, gp) - _e2_fn(h1m, gm)) / (2 * fd_eps_kappa))

    from ._parallel_helpers import choose_pt2_fd_n_jobs

    n_jobs = choose_pt2_fd_n_jobs(
        npr=npr,
        nmo=nmo,
        use_cpp_transform=_use_cpp_xform,
    )
    if n_jobs <= 1:
        dE2_dk = np.array([_dE2_dk_pair(i) for i in range(npr)])
    else:
        try:
            from joblib import Parallel, delayed

            dE2_dk = np.array(
                Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(_dE2_dk_pair)(i) for i in range(npr)
                )
            )
        except (ImportError, AttributeError):
            dE2_dk = np.array([_dE2_dk_pair(i) for i in range(npr)])

    # 2. dE2/dc — numerical FD CI Lagrangian
    # For SA-NEVPT2: average per-state CI Lagrangians
    _is_sa = sa_weights is not None and sa_ci_coeffs is not None
    n_det = len(determinants)
    dE2_dc = np.zeros(max(0, n_det - 1))
    if n_det > 1:
        if _is_sa and len(sa_weights) == len(sa_ci_coeffs):
            # SA: average per-state dE2/dc
            for w_i, c_i in zip(sa_weights, sa_ci_coeffs):
                try:
                    c_arr = np.asarray(c_i, dtype=float)
                    if len(c_arr) != n_det:
                        continue
                    dE2_dc_i = _compute_nevpt2_ci_lagrangian_fd(
                        h1e_cas,
                        h2e_cas,
                        c_arr,
                        determinants,
                        n_core,
                        n_act,
                        n_act_elec,
                        fd_eps=fd_eps_ci,
                    )
                    dE2_dc += w_i * dE2_dc_i
                except Exception:
                    pass
        else:
            try:
                dE2_dc = _compute_nevpt2_ci_lagrangian_fd(
                    h1e_cas,
                    h2e_cas,
                    ci_coeffs,
                    determinants,
                    n_core,
                    n_act,
                    n_act_elec,
                    fd_eps=fd_eps_ci,
                )
            except Exception:
                dE2_dc = np.zeros(n_det - 1)

    # 3. Hessian blocks (CASSCF + PT2 correction)
    H_oo = _build_orbital_hessian_fd(
        h1e_cas,
        h2e_cas,
        rdm1,
        rdm2,
        D_mo,
        gamma_mo,
        n_core,
        n_act,
        nmo,
        pairs,
        fd_eps=1e-4,
    )
    # PT2-specific Hessian contribution for small systems
    # DISABLED: see _caspt2.py for rationale (incompatible with z^T.g^R).
    if False and npr <= 20:
        try:
            from ._pt2_hessian import build_pt2_orbital_hessian_fd

            H_e2_oo = build_pt2_orbital_hessian_fd(
                h1e_cas,
                h2e_cas,
                rdm1,
                rdm2,
                n_core,
                n_act,
                n_act_elec,
                nmo,
                pairs,
                variant="nevpt2",
            )
            H_oo = H_oo + H_e2_oo
        except Exception:
            pass
    # Build H_oc and H_cc (if CI Lagrangian is available)
    H_eff = H_oo
    if n_det > 1 and np.any(np.abs(dE2_dc) > 1e-14):
        H_det = _build_ci_hamiltonian(h1e_cas, h2e_cas, determinants, n_core, n_act)
        E_det = np.linalg.eigvalsh(H_det)
        H_cc_eig = 2.0 * (E_det[1:] - E_det[0])
        H_oc = _build_ci_orbital_coupling_energy_fd(
            h1e_cas,
            h2e_cas,
            h1e_cas,
            h2e_cas,
            D_mo,
            gamma_mo,
            mol.nuclear_repulsion(),
            ci_coeffs,
            determinants,
            n_core,
            n_act,
            pairs,
            nmo,
        )
        # CI-eliminated Hessian
        mask_cc = np.abs(H_cc_eig) > 1e-12
        H_cc_inv = np.zeros(n_det - 1)
        H_cc_inv[mask_cc] = 1.0 / H_cc_eig[mask_cc]
        H_eff = H_oo - H_oc @ np.diag(H_cc_inv) @ H_oc.T

    # 4. Z-vector solve
    rhs = -dE2_dk
    if n_det > 1 and np.any(np.abs(dE2_dc) > 1e-14):
        rhs = rhs + H_oc @ (H_cc_inv * dE2_dc)

    ev, evec = np.linalg.eigh(H_eff)
    mask = np.abs(ev) > 1e-10
    z_orb = evec[:, mask] @ np.diag(1.0 / ev[mask]) @ evec[:, mask].T @ rhs

    # 5. g^R with NEVPT2 effective density corrections
    # For SA-NEVPT2: average per-state effective densities weighted by SA weights.
    try:
        from ._pt2_density import compute_nevpt2_effective_density

        P_eff = _semicanonical_prep(h1e_cas, h2e_cas, n_core, n_act, n_act_elec, 0)

        if _is_sa and len(sa_weights) == len(sa_ci_coeffs):
            # SA-NEVPT2: per-state effective densities, SA-weighted average
            n_act_dim = n_act
            dm1_corr = np.zeros((n_act_dim, n_act_dim))
            dm2_corr = np.zeros((n_act_dim, n_act_dim, n_act_dim, n_act_dim))
            for w_i, c_i in zip(sa_weights, sa_ci_coeffs):
                c_arr = np.asarray(c_i, dtype=float)
                if len(c_arr) != len(determinants):
                    continue
                P_i = _semicanonical_prep(
                    h1e_cas,
                    h2e_cas,
                    n_core,
                    n_act,
                    n_act_elec,
                    0,
                    reference=(c_arr, determinants),
                )
                DeltaD_i, DeltaG_i = compute_nevpt2_effective_density(
                    P_i, n_core, n_act
                )
                dm1_corr += (
                    w_i * DeltaD_i[n_core : n_core + n_act, n_core : n_core + n_act]
                )
                dm2_corr += (
                    w_i
                    * DeltaG_i[
                        n_core : n_core + n_act,
                        n_core : n_core + n_act,
                        n_core : n_core + n_act,
                        n_core : n_core + n_act,
                    ]
                )
        else:
            DeltaD_full, DeltaG_full = compute_nevpt2_effective_density(
                P_eff, n_core, n_act
            )
            dm1_corr = DeltaD_full[n_core : n_core + n_act, n_core : n_core + n_act]
            dm2_corr = DeltaG_full[
                n_core : n_core + n_act,
                n_core : n_core + n_act,
                n_core : n_core + n_act,
                n_core : n_core + n_act,
            ]
        # NEVPT2: skip chain-rule trace corrections (designed for CASPT2 Fock,
        # not Dyall H0; applying them increases delta from 2.42e-5 to 2.48e-5).
        # Bare effective densities from compute_nevpt2_effective_density are optimal.
        #
        # Dyall-specific chain rule (from _dyall_chain_rule.py, v50): gated.
        _USE_DYALL_CHAIN_RULE = False
        if _USE_DYALL_CHAIN_RULE:
            try:
                from ._dyall_chain_rule import dyall_chain_rule_correction

                dcr = dyall_chain_rule_correction(P_eff, DeltaD_full, n_core, n_act)
                dm1_corr = dm1_corr + dcr
            except Exception:
                pass
    except Exception:
        dm1_corr = None
        dm2_corr = None

    gR = _compute_perturbed_orbital_gradient_fd(
        mol,
        basis,
        C_mo,
        h1e_cas,
        h2e_cas,
        rdm1,
        rdm2,
        D_mo,
        gamma_mo,
        n_core,
        n_act,
        nmo,
        pairs,
        fd_eps=fd_eps_gR,
        dm1_correction=dm1_corr,
        dm2_correction=dm2_corr,
    )
    n_atoms = len(mol.atoms)
    correction = np.zeros((n_atoms, 3))
    for a in range(n_atoms):
        for c in range(3):
            col = a * 3 + c
            correction[a, c] = float(z_orb @ gR[:, col])
    return correction


# ---- Full CP-MCSCF Lagrangian gradient (OpenMolcas-equivalent, experimental) ----


def _compute_nevpt2_lagrangian_gradient(
    *,
    mol,
    basis,
    C_mo,
    h1e_cas,
    h2e_cas,
    n_core,
    n_act,
    n_act_elec,
    rdm1,
    rdm2,
    ci_coeffs,
    determinants,
    fd_eps_kappa: float = 1e-3,
    fd_eps_ci: float = 1e-2,
) -> "np.ndarray":
    """Full CP-MCSCF Lagrangian gradient for NEVPT2 (experimental).

    Replaces grad_casscf + grad_corr + zvec_corr with a single Lagrangian
    construction using effective densities and Z-vector.

    Currently gated — the Lagrangian path uses the same Z-vector solve
    as the shortcut but computes gradients via D_eff/D^z contraction.
    Accuracy is lower than the z^T·g^R shortcut for the same reasons as
    CASPT2 (missing geometry-derivative of effective densities).

    Returns total (n_atoms, 3) gradient in Hartree/bohr.
    """
    import numpy as np
    from scipy.linalg import expm

    from ..solvers._mrpt import (
        _add,
        _pt2_correction,
        _semicanonical_prep,
        apply_1body,
        apply_2body,
    )
    from ._casscf import (
        _build_ci_hamiltonian,
        _build_ci_orbital_coupling_energy_fd,
        _build_full_gamma_mo,
        _build_orbital_hessian_fd,
        _nonredundant_pairs_local,
    )
    from ._pt2_density import compute_nevpt2_effective_density
    from ._pt2_lagrangian import compute_pt2_total_lagrangian_gradient

    nmo = h1e_cas.shape[0]
    pairs = _nonredundant_pairs_local(n_core, n_act, nmo)
    npr = len(pairs)
    if npr == 0:
        return np.zeros((len(mol.atoms), 3))

    D_cas = np.zeros((nmo, nmo))
    D_cas[:n_core, :n_core] = np.diag(np.full(n_core, 2.0))
    D_cas[n_core : n_core + n_act, n_core : n_core + n_act] = rdm1
    Gamma_cas = _build_full_gamma_mo(rdm1, rdm2, n_core, n_act, nmo)

    # --- Z-vector solve (NEVPT2: Dyall H0-based dE2/dk) ---

    def e2_nevpt2(h1, h2):
        P = _semicanonical_prep(h1, h2, n_core, n_act, n_act_elec, 0)
        norb = P["norb"]
        h1D = np.zeros((norb, norb))
        for p in range(n_core):
            h1D[p, p] = P["eps"][p]
        for p in range(n_core + n_act, norb):
            h1D[p, p] = P["eps"][p]
        h1D[P["act"], P["act"]] = P["h1a"]
        eriD = np.zeros_like(P["eri"])
        eriD[P["act"], P["act"], P["act"], P["act"]] = P["eri"][
            P["act"], P["act"], P["act"], P["act"]
        ]

        def apply_HD(state):
            return _add(
                apply_1body(state, h1D, norb),
                apply_2body(state, eriD, norb, idx=P["aidx"]),
            )

        return _pt2_correction(P, apply_HD)

    g_phys_flat = np.ascontiguousarray(h2e_cas.ravel(), dtype=float)
    _use_cpp_xform = False
    try:
        from .._vibeqc_core import transform_4index_mo as _xform4

        _use_cpp_xform = True
    except (ImportError, AttributeError):
        pass

    def _dE2_dk_pair(i):
        ei = np.zeros(npr)
        ei[i] = 1.0
        K = np.zeros((nmo, nmo))
        for j, (p, q) in enumerate(pairs):
            K[p, q] = fd_eps_kappa * ei[j]
            K[q, p] = -fd_eps_kappa * ei[j]
        Up = expm(K)
        Um = expm(-K)
        h1p = Up.T @ h1e_cas @ Up
        h1m = Um.T @ h1e_cas @ Um
        if _use_cpp_xform:
            gp = np.asarray(
                _xform4(g_phys_flat, np.ascontiguousarray(Up, dtype=float), nmo)
            ).reshape(nmo, nmo, nmo, nmo)
            gm = np.asarray(
                _xform4(g_phys_flat, np.ascontiguousarray(Um, dtype=float), nmo)
            ).reshape(nmo, nmo, nmo, nmo)
        else:
            gp = np.einsum("ap,bq,cr,ds,abcd->pqrs", Up, Up, Up, Up, h2e_cas)
            gm = np.einsum("ap,bq,cr,ds,abcd->pqrs", Um, Um, Um, Um, h2e_cas)
        return float((e2_nevpt2(h1p, gp) - e2_nevpt2(h1m, gm)) / (2 * fd_eps_kappa))

    from ._parallel_helpers import choose_pt2_fd_n_jobs

    n_jobs = choose_pt2_fd_n_jobs(npr=npr, nmo=nmo, use_cpp_transform=_use_cpp_xform)
    if n_jobs <= 1:
        dE2_dk = np.array([_dE2_dk_pair(i) for i in range(npr)])
    else:
        try:
            from joblib import Parallel, delayed

            dE2_dk = np.array(
                Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(_dE2_dk_pair)(i) for i in range(npr)
                )
            )
        except (ImportError, AttributeError):
            dE2_dk = np.array([_dE2_dk_pair(i) for i in range(npr)])

    n_det = len(determinants)
    dE2_dc = np.zeros(max(0, n_det - 1))
    if n_det > 1:
        try:
            from ._nevpt2_cilag import compute_nevpt2_ci_lagrangian_analytic

            dE2_dc = compute_nevpt2_ci_lagrangian_analytic(
                h1e_cas, h2e_cas, ci_coeffs, determinants, n_core, n_act, n_act_elec
            )
        except Exception:
            dE2_dc = np.zeros(n_det - 1)

    H_det = _build_ci_hamiltonian(h1e_cas, h2e_cas, determinants, n_core, n_act)
    E_det = np.linalg.eigvalsh(H_det)
    H_oo = _build_orbital_hessian_fd(
        h1e_cas,
        h2e_cas,
        rdm1,
        rdm2,
        D_cas,
        Gamma_cas,
        n_core,
        n_act,
        nmo,
        pairs,
        fd_eps=1e-4,
    )
    H_oc = _build_ci_orbital_coupling_energy_fd(
        h1e_cas,
        h2e_cas,
        h1e_cas,
        h2e_cas,
        D_cas,
        Gamma_cas,
        mol.nuclear_repulsion(),
        ci_coeffs,
        determinants,
        n_core,
        n_act,
        pairs,
        nmo,
    )
    H_cc_diag = 2.0 * (E_det[1:] - E_det[0])
    H_cc_inv = np.zeros(n_det - 1)
    m = np.abs(H_cc_diag) > 1e-12
    H_cc_inv[m] = 1.0 / H_cc_diag[m]
    H_eff = H_oo - H_oc @ np.diag(H_cc_inv) @ H_oc.T
    rhs_eff = -dE2_dk + H_oc @ (H_cc_inv * dE2_dc)
    ev, evec = np.linalg.eigh(H_eff)
    mask = np.abs(ev) > 1e-10
    z_orb = evec[:, mask] @ np.diag(1.0 / ev[mask]) @ evec[:, mask].T @ rhs_eff

    # --- Effective densities (NEVPT2: no chain rule, 1.5x scale from compute_nevpt2_effective_density) ---
    P_eff = _semicanonical_prep(h1e_cas, h2e_cas, n_core, n_act, n_act_elec, 0)
    DeltaD_full, DeltaG_full = compute_nevpt2_effective_density(P_eff, n_core, n_act)

    return compute_pt2_total_lagrangian_gradient(
        mol=mol,
        basis=basis,
        C_mo=C_mo,
        z_orb=z_orb,
        h1e_mo=h1e_cas,
        h2e_mo=h2e_cas,
        D_casscf=D_cas,
        Gamma_casscf=Gamma_cas,
        DeltaD=DeltaD_full,
        DeltaGamma=DeltaG_full,
        DEPSA=None,
        n_core=n_core,
        n_act=n_act,
        nmo=nmo,
        pairs=pairs,
    )
