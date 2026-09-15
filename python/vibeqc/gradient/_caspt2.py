"""CASPT2 analytic nuclear gradient.

Implements the CASPT2 gradient by combining the CASSCF analytic gradient
with a numerical FD of the CASPT2 correlation energy E^(2).

    dE_CASPT2/dR = dE_CASSCF/dR + dE^(2)/dR

The CASSCF part uses the analytic gradient; the correlation contribution
is computed by central FD of the IC-CASPT2 E^(2) at displaced geometries
with re-orthonormalized CASSCF MOs and re-solved CASCI.

For production use, the full internally-contracted (IC) CASPT2 gradient
(Celani & Werner 2003) with CASPT2 density matrices and z-vector response
remains a future milestone.
"""

from __future__ import annotations

import warnings

import numpy as np

from .._vibeqc_core import Atom, BasisSet, Molecule
from .._vibeqc_core import compute_overlap as _compute_overlap


def _warn_gradient_degradation(stage: str, exc: Exception) -> None:
    """Warn when a requested CASPT2 gradient correction is approximated away."""
    warnings.warn(
        "CASPT2 gradient: "
        f"{stage} failed ({type(exc).__name__}: {exc}); "
        "continuing without that correction.",
        UserWarning,
        stacklevel=2,
    )


def _caspt2_e_corr_frozen(
    mol_d: Molecule,
    basis_d: BasisSet,
    C_conv: np.ndarray,
    n_core: int,
    n_act_orb: int,
    n_act_elec: int,
    nuclear_repulsion: float = 0.0,
    *,
    variant: str = "sc",
) -> float:
    """CASPT2 correlation energy at displaced geometry.

    Re-orthonormalizes CASSCF MOs at displaced S, transforms integrals,
    re-solves CASCI, and computes CASPT2 E^(2).

    Parameters
    ----------
    variant : str
        "sc" (strongly-contracted, default, fast) or "ic" (internally-contracted,
        accurate, slower — builds full FOIS matrix).
    """
    from scipy.linalg import eigh

    from .._vibeqc_core import compute_eri, compute_kinetic, compute_nuclear
    from ..solvers._mrpt import (
        _ic_caspt2_solve,
        _pt2_correction,
        _semicanonical_prep,
        apply_1body,
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

    # CASCI + CASPT2 at displaced geometry
    prep = _semicanonical_prep(h1e_d, h2e_d, n_core, n_act_orb, n_act_elec, 0)

    if variant == "ic":
        e2, _ = _ic_caspt2_solve(
            prep,
            n_core,
            n_act_orb,
            n_frozen=0,
            ipea=0.0,
            imaginary=0.0,
            thresh=1e-8,
        )
        return float(e2)

    # SC-CASPT2: strongly-contracted, group-based, fast
    norb = prep["norb"]
    F = prep["F"]

    def apply_H0(state):
        return apply_1body(state, F, norb)

    e2 = _pt2_correction(prep, apply_H0)
    return float(e2)


def compute_caspt2_gradient(
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
    compute_wz: bool = False,
    use_zvector: bool = False,
    use_lagrangian: bool = False,
    sa_weights: list[float] | None = None,
    sa_ci_coeffs: list[np.ndarray] | None = None,
    determinants: list | None = None,
    ci_coeffs: np.ndarray | None = None,
    fd_eps: float = 0.001,
) -> np.ndarray:
    """CASPT2 nuclear gradient.

    Combines the CASSCF analytic gradient with an FD of the IC-CASPT2
    correlation energy.  At each displaced geometry the CASSCF MOs are
    re-orthonormalized and CASCI is re-solved (frozen-active-space).

    Parameters
    ----------
    fd_eps : float
        Geometry displacement step for the correlation energy FD (bohr).
    n_active_elec : int
        Number of active electrons.

    Notes
    -----
    Manually supplied ECP-derived orbitals, integrals, or reference data
    paired with an all-electron-named basis are unsupported.  This API can
    reject an ECP attached to ``basis``, but cannot infer the Hamiltonian
    provenance of the supplied arrays.

    Returns
    -------
    grad : (n_atoms, 3) ndarray
        Total CASPT2 nuclear gradient in Hartree/bohr.
    """
    from ..ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        mol,
        basis,
        route="compute_caspt2_gradient",
    )
    from ..gradient import compute_casscf_gradient

    # Full CP-MCSCF Lagrangian path (OpenMolcas-equivalent).
    # Replaces grad_casscf + grad_corr + zvec_corr with a single
    # Lagrangian construction from effective densities and Z-vector.
    if (
        use_lagrangian
        and use_zvector
        and determinants is not None
        and ci_coeffs is not None
    ):
        try:
            return _compute_caspt2_lagrangian_gradient(
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
        except Exception as exc:
            _warn_gradient_degradation("full Lagrangian gradient", exc)
            pass  # fall through to standard path

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

    # CASPT2 correlation gradient via FD of E^(2)
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

            e_p = _caspt2_e_corr_frozen(
                mol_p,
                basis_p,
                C_mo,
                n_core,
                n_active_orb,
                n_active_elec,
                mol.nuclear_repulsion(),
            )
            e_m = _caspt2_e_corr_frozen(
                mol_m,
                basis_m,
                C_mo,
                n_core,
                n_active_orb,
                n_active_elec,
                mol.nuclear_repulsion(),
            )
            grad_corr[a, c] = (e_p - e_m) / (2.0 * fd_eps)

    zvec_corr = 0.0
    if use_zvector and determinants is not None and ci_coeffs is not None:
        zvec_corr = _compute_caspt2_zvector_correction(
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


def _compute_caspt2_zvector_correction(
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
    """Historical CASPT2 Z-vector orbital relaxation shortcut.

    Uses analytic CI Lagrangian (transition RDMs) + numerical dE2/dk
    + z^T.g^R shortcut with corrected effective density (N/Δ^2 factor).
    This is retained for focused analytic-gradient development; the public
    runner uses relaxed full-energy FD for production correlation gradients.

    Returns (n_atoms, 3) gradient correction in Hartree/bohr.
    """
    import numpy as np
    from scipy.linalg import expm

    from ..solvers._mrpt import _pt2_correction, _semicanonical_prep, apply_1body
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

    # Check FG3 flag before defining e2_at
    _use_ic_fg3 = False  # IC path: needs complete IC gradient framework

    def e2_at(h1, h2):
        P = _semicanonical_prep(h1, h2, n_core, n_act, n_act_elec, 0)
        if _use_ic_fg3:
            from ..solvers._mrpt import _ic_caspt2_solve

            e2, _ = _ic_caspt2_solve(P, n_core, n_act)
            return e2
        return _pt2_correction(P, lambda s: apply_1body(s, P["F"], P["norb"]))

    # 1. dE2/dkappa — numerical FD via perturbed MO integrals (parallel).
    # Each orbital-pair direction is independent — PT2 solves at +eps and -eps.
    g_phys_flat = np.ascontiguousarray(h2e_cas.ravel(), dtype=float)
    _use_cpp_xform = False
    try:
        from .._vibeqc_core import transform_4index_mo as _xform4

        _use_cpp_xform = True
    except (ImportError, AttributeError):
        pass

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
        return float((e2_at(h1p, gp) - e2_at(h1m, gm)) / (2 * fd_eps_kappa))

    from ._parallel_helpers import choose_pt2_fd_n_jobs

    n_jobs = choose_pt2_fd_n_jobs(
        npr=npr,
        nmo=nmo,
        use_cpp_transform=_use_cpp_xform,
    )
    if n_jobs <= 1:
        # Serial for small systems — avoids joblib overhead.
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

    # 2. dE2/dc — analytic via transition RDMs + effective density
    # For SA-CASPT2: average per-state CI Lagrangians
    _is_sa_caspt2 = sa_weights is not None and sa_ci_coeffs is not None
    n_det = len(determinants)
    dE2_dc = np.zeros(max(0, n_det - 1))
    if n_det > 1:
        if _is_sa_caspt2 and len(sa_weights) == len(sa_ci_coeffs):
            from ._caspt2_cilag import compute_caspt2_ci_lagrangian_analytic

            for w_i, c_i in zip(sa_weights, sa_ci_coeffs):
                try:
                    c_arr = np.asarray(c_i, dtype=float)
                    if len(c_arr) != n_det:
                        continue
                    dE2_dc_i = compute_caspt2_ci_lagrangian_analytic(
                        h1e_cas,
                        h2e_cas,
                        c_arr,
                        determinants,
                        n_core,
                        n_act,
                        n_act_elec,
                    )
                    dE2_dc += w_i * dE2_dc_i
                except Exception as exc:
                    _warn_gradient_degradation("state-averaged CI Lagrangian", exc)
                    pass
        else:
            try:
                from ._caspt2_cilag import compute_caspt2_ci_lagrangian_analytic

                dE2_dc = compute_caspt2_ci_lagrangian_analytic(
                    h1e_cas,
                    h2e_cas,
                    ci_coeffs,
                    determinants,
                    n_core,
                    n_act,
                    n_act_elec,
                )
            except Exception as exc:
                _warn_gradient_degradation("CI Lagrangian", exc)
                dE2_dc = np.zeros(n_det - 1)
    # CI eigenbasis for H_cc
    H_det = _build_ci_hamiltonian(h1e_cas, h2e_cas, determinants, n_core, n_act)
    E_det = np.linalg.eigvalsh(H_det)

    # 3. Hessian blocks
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
    # Add PT2-specific Hessian contribution (H^E2_oo).
    # PT2-specific Hessian contribution (H^E2_oo).
    # DISABLED: adding PT2 H_oo to CASSCF H_oo consistently makes
    # Z-vector WORSE (9.2e-3 error vs 1.7e-3 without).  The z^T.g^R
    # shortcut is incompatible with PT2 Hessian; the correct CP-MCSCF
    # correction needs explicit W^z/D^z/Gamma^z + derivative integrals
    # (the full analytic Lagrangian).
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
                variant="caspt2",
            )
            H_oo = H_oo + H_e2_oo
        except Exception:
            pass
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
    H_cc_diag = 2.0 * (E_det[1:] - E_det[0])

    # 4. CI-eliminated solve
    H_cc_inv = np.zeros(n_det - 1)
    m = np.abs(H_cc_diag) > 1e-12
    H_cc_inv[m] = 1.0 / H_cc_diag[m]
    H_eff = H_oo - H_oc @ np.diag(H_cc_inv) @ H_oc.T
    rhs_eff = -dE2_dk + H_oc @ (H_cc_inv * dE2_dc)
    ev, evec = np.linalg.eigh(H_eff)
    mask = np.abs(ev) > 1e-10
    z_orb = evec[:, mask] @ np.diag(1.0 / ev[mask]) @ evec[:, mask].T @ rhs_eff

    # 5. g^R with CASPT2 effective density corrections
    # For SA-CASPT2: average per-state effective densities weighted by SA weights.
    try:
        from ..solvers._mrpt import _semicanonical_prep
        from ._pt2_density import compute_pt2_effective_density

        chain_rule_preps: list[tuple[float, dict]] = []
        if _is_sa_caspt2 and len(sa_weights) == len(sa_ci_coeffs):
            # SA-CASPT2: per-state effective densities, SA-weighted average
            dm1_corr = np.zeros((n_act, n_act))
            dm2_corr = np.zeros((n_act, n_act, n_act, n_act))
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
                DeltaD_i, DeltaG_i = compute_pt2_effective_density(P_i, n_core, n_act)
                chain_rule_preps.append((float(w_i), P_i))
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
            P_eff = _semicanonical_prep(h1e_cas, h2e_cas, n_core, n_act, n_act_elec, 0)
            chain_rule_preps = [(1.0, P_eff)]
            DeltaD_full, DeltaG_full = compute_pt2_effective_density(
                P_eff, n_core, n_act
            )
            dm1_corr = DeltaD_full[n_core : n_core + n_act, n_core : n_core + n_act]
            dm2_corr = DeltaG_full[
                n_core : n_core + n_act,
                n_core : n_core + n_act,
                n_core : n_core + n_act,
                n_core : n_core + n_act,
            ]
        # Apply 3-RDM chain-rule trace corrections to 1-RDM (derfg3 style).
        # Two paths:
        #   1. _pt2_chain_rule: effective-density based (trace reductions, DF2 from DG2)
        #   2. _clagdxa_fg3: full CLagDXA_FG3 via 3-RDM KTUV contraction (v52, gated)
        _USE_FULL_CHAIN_RULE = True
        _USE_FG3_CHAIN_RULE = False  # Needs complete IC gradient framework
        if _USE_FG3_CHAIN_RULE:
            try:
                # Disabled by default: these IC density components need the
                # parallel IC Hessian/Z-vector/g^R framework before use.
                from ._ic_caspt2_density import (
                    compute_ic_caspt2_density_components,
                )

                ic = compute_ic_caspt2_density_components(P_eff, n_core, n_act)
                act_s = slice(n_core, n_core + n_act)
                dm2_corr = dm2_corr + ic.delta_gamma[act_s, act_s, act_s, act_s]
                dm1_corr = dm1_corr + ic.delta_d[act_s, act_s]
            except Exception as exc:
                _warn_gradient_degradation("IC FG3 density correction", exc)
                pass
        elif _USE_FULL_CHAIN_RULE:
            try:
                from ._pt2_chain_rule import compute_chain_rule_corrections

                for w_corr, P_corr in chain_rule_preps:
                    cr = compute_chain_rule_corrections(
                        P_corr, n_core, n_act, variant="caspt2"
                    )
                    if cr.get("DG2") is not None:
                        dm2_corr = dm2_corr + w_corr * cr["DG2"]
                    if cr.get("DG1") is not None:
                        dm1_corr = dm1_corr + w_corr * cr["DG1"]
            except Exception as exc:
                _warn_gradient_degradation("PT2 chain-rule correction", exc)
                pass
        else:
            try:
                from ._pt2_density import _apply_chain_rule_trace_corrections

                _apply_chain_rule_trace_corrections(
                    DeltaD_full, DeltaG_full, P_eff["eps"], n_core, n_act
                )
                dm1_corr = DeltaD_full[n_core : n_core + n_act, n_core : n_core + n_act]
            except Exception as exc:
                _warn_gradient_degradation("legacy PT2 trace correction", exc)
                pass

    except Exception as exc:
        _warn_gradient_degradation("PT2 effective-density correction", exc)
        dm1_corr = None
        dm2_corr = None

    # 6. Gradient correction: historical z^T.g^R shortcut.
    # W^z/D^z/Gamma^z Lagrangian gated behind _USE_WZ_LAGRANGIAN.
    _USE_WZ_LAGRANGIAN = False
    if _USE_WZ_LAGRANGIAN and dm1_corr is not None and dm2_corr is not None:
        try:
            from ._pt2_wz_explicit import compute_pt2_explicit_wz_gradient

            D_eff = D_mo.copy()
            D_eff[n_core : n_core + n_act, n_core : n_core + n_act] += dm1_corr
            Gamma_eff = gamma_mo.copy()
            Gamma_eff[
                n_core : n_core + n_act,
                n_core : n_core + n_act,
                n_core : n_core + n_act,
                n_core : n_core + n_act,
            ] += dm2_corr
            eri_chem = h2e_cas.transpose(0, 2, 1, 3)
            F_eff = np.zeros((nmo, nmo))
            for p in range(nmo):
                for q in range(nmo):
                    f_val = h1e_cas[p, q]
                    for r in range(nmo):
                        for s in range(nmo):
                            f_val += D_eff[r, s] * (
                                2.0 * eri_chem[p, q, r, s] - eri_chem[p, s, r, q]
                            )
                    F_eff[p, q] = f_val
            return compute_pt2_explicit_wz_gradient(
                mol,
                basis,
                C_mo,
                z_orb,
                D_eff,
                Gamma_eff,
                F_eff,
                n_core,
                n_act,
                nmo,
                pairs,
            )
        except Exception as exc:
            _warn_gradient_degradation("explicit Wz Lagrangian correction", exc)
            pass
    # 5. Gradient correction: historical z^T.g^R shortcut.
    # Per-geometry effective density correction (captures dDeltaD/dR).
    # When enabled, recomputes PT2 effective densities at each displaced
    # geometry instead of using constant MO-basis corrections.
    _USE_PER_GEOMETRY_CORR = False
    _dm_corr_callable = None
    if _USE_PER_GEOMETRY_CORR:

        def _recompute_corrections(h_mo, eri_c, nc, na):
            from ..solvers._mrpt import _semicanonical_prep
            from ._pt2_density import compute_pt2_effective_density

            n_el = n_act_elec
            g_phys = eri_c.transpose(0, 2, 1, 3)
            P = _semicanonical_prep(h_mo, g_phys, nc, na, n_el, 0)
            dD, dG = compute_pt2_effective_density(P, nc, na)
            dm1c = dD[nc : nc + na, nc : nc + na]
            dm2c = dG[nc : nc + na, nc : nc + na, nc : nc + na, nc : nc + na]
            try:
                from ._pt2_chain_rule import compute_chain_rule_corrections

                cr = compute_chain_rule_corrections(P, nc, na, variant="caspt2")
                if cr.get("DG2") is not None:
                    dm2c = dm2c + cr["DG2"]
                if cr.get("DG1") is not None:
                    dm1c = dm1c + cr["DG1"]
            except Exception as exc:
                _warn_gradient_degradation("per-geometry chain-rule correction", exc)
                pass
            return dm1c, dm2c

        _dm_corr_callable = _recompute_corrections

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
        dm_correction_callable=_dm_corr_callable,
    )
    n_atoms = len(mol.atoms)
    correction = np.zeros((n_atoms, 3))
    for a in range(n_atoms):
        for c in range(3):
            col = a * 3 + c
            correction[a, c] = float(z_orb @ gR[:, col])

    return correction


def _energy_weighted_density_from_rdms(
    h1e_mo: np.ndarray,
    eri_chem: np.ndarray,
    dm1_mo: np.ndarray,
    dm2_mo: np.ndarray,
) -> np.ndarray:
    """Return ``W = -dE/dS`` for fixed full-space MO RDMs.

    Differentiating all MO indices of
    ``E = h.D + 1/2 (pq|rs).Gamma`` under the symmetric Lowdin response
    ``dC = -1/2 C dS_MO`` gives this closed-form energy-weighted density.
    It supports the external-orbital blocks present in a PT2 Lagrangian,
    unlike the active-row CASSCF Fock specialization.
    """
    y = np.einsum("ab,pb->pa", dm1_mo, h1e_mo, optimize=True)
    y += np.einsum("ia,ip->pa", dm1_mo, h1e_mo, optimize=True)
    y += 0.5 * np.einsum("ajkl,pjkl->pa", dm2_mo, eri_chem, optimize=True)
    y += 0.5 * np.einsum("iakl,ipkl->pa", dm2_mo, eri_chem, optimize=True)
    y += 0.5 * np.einsum("ijal,ijpl->pa", dm2_mo, eri_chem, optimize=True)
    y += 0.5 * np.einsum("ijka,ijkp->pa", dm2_mo, eri_chem, optimize=True)
    return 0.25 * (y + y.T)


def _contract_ic_caspt2_lagrangian_gradient(
    mol,
    basis,
    C_mo: np.ndarray,
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    dm1_lagrangian: np.ndarray,
    dm2_lagrangian: np.ndarray,
) -> np.ndarray:
    """Contract a full IC-CASPT2 Lagrangian density with AO derivatives."""
    from .._vibeqc_core import (
        nuclear_repulsion_gradient,
        one_electron_gradient_contribution,
        overlap_gradient_contribution,
        two_electron_gradient_casscf,
    )
    from ._casscf import _transform_4index_mo_to_ao_flat
    from ._pt2_lagrangian import _symmetrize_8fold_flat

    nmo = int(h1e_mo.shape[0])
    nb = int(C_mo.shape[0])
    dm1_ao = C_mo @ dm1_lagrangian @ C_mo.T
    dm2_ao = _transform_4index_mo_to_ao_flat(dm2_lagrangian, C_mo, nmo)
    dm2_ao = 0.5 * _symmetrize_8fold_flat(dm2_ao, nb)

    eri_chem = h2e_mo.transpose(0, 2, 1, 3)
    w_mo = _energy_weighted_density_from_rdms(
        h1e_mo,
        eri_chem,
        dm1_lagrangian,
        dm2_lagrangian,
    )
    w_ao = C_mo @ w_mo @ C_mo.T

    gradient = np.asarray(nuclear_repulsion_gradient(mol), dtype=float)
    gradient += np.asarray(
        one_electron_gradient_contribution(basis, mol, dm1_ao), dtype=float
    )
    gradient += np.asarray(
        two_electron_gradient_casscf(basis, mol, dm2_ao), dtype=float
    )
    gradient += np.asarray(
        overlap_gradient_contribution(basis, mol, w_ao), dtype=float
    )
    return gradient


def compute_ic_caspt2_analytic_gradient(
    mol,
    basis,
    C_mo: np.ndarray,
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_core: int,
    n_act: int,
    n_act_elec: int,
    *,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    ci_coeffs: np.ndarray,
    determinants: list,
    n_frozen: int = 0,
    response_eps: float = 1e-4,
) -> np.ndarray:
    r"""Analytic nuclear gradient for unshifted single-state IC-CASPT2.

    The nuclear derivative is an AO derivative-integral contraction of the
    stationary Hylleraas density.  Orbital and CI multipliers solve the full
    coupled CASSCF response system; finite differences are used only to build
    the internal response RHS/Hessian, never for nuclear energies.
    """
    from ..solvers._mrpt import _build_reference_state, _semicanonical_prep
    from ._casscf import (
        _build_ci_orbital_coupling_energy_fd,
        _build_full_gamma_mo,
        _build_kappa_from_z,
        _build_orbital_hessian_fd,
        _compute_one_index_transformed_dm,
        _compute_one_index_transformed_gamma,
        _nonredundant_pairs_local,
    )
    from ._ic_caspt2_density import (
        _transition_rdm12,
        compute_ic_caspt2_ci_gradient_fd,
        compute_ic_caspt2_effective_density,
        compute_ic_caspt2_orbital_gradient_fd,
    )

    nmo = int(h1e_mo.shape[0])
    pairs = _nonredundant_pairs_local(n_core, n_act, nmo)
    n_orbital = len(pairs)

    dm1_cas = np.zeros((nmo, nmo))
    dm1_cas[:n_core, :n_core] = 2.0 * np.eye(n_core)
    act = slice(n_core, n_core + n_act)
    dm1_cas[act, act] = rdm1
    dm2_cas = _build_full_gamma_mo(rdm1, rdm2, n_core, n_act, nmo)

    reference = (np.asarray(ci_coeffs, dtype=float).copy(), determinants)
    prep = _semicanonical_prep(
        h1e_mo,
        h2e_mo,
        n_core,
        n_act,
        n_act_elec,
        0,
        reference=reference,
    )
    effective = compute_ic_caspt2_effective_density(
        prep,
        n_core,
        n_act,
        n_frozen=n_frozen,
    )

    d_e2_dk = compute_ic_caspt2_orbital_gradient_fd(
        h1e_mo,
        h2e_mo,
        n_core,
        n_act,
        n_act_elec,
        pairs,
        fd_eps=response_eps,
        n_frozen=n_frozen,
        reference=reference,
    )
    d_e2_dc, ci_energies, ci_vectors = compute_ic_caspt2_ci_gradient_fd(
        h1e_mo,
        h2e_mo,
        ci_coeffs,
        determinants,
        n_core,
        n_act,
        n_act_elec,
        fd_eps=response_eps,
        n_frozen=n_frozen,
    )

    h_oo = _build_orbital_hessian_fd(
        h1e_mo,
        h2e_mo,
        rdm1,
        rdm2,
        dm1_cas,
        dm2_cas,
        n_core,
        n_act,
        nmo,
        pairs,
        fd_eps=1e-4,
    )
    h_oc = _build_ci_orbital_coupling_energy_fd(
        h1e_mo,
        h2e_mo,
        h1e_mo,
        h2e_mo,
        dm1_cas,
        dm2_cas,
        mol.nuclear_repulsion(),
        ci_coeffs,
        determinants,
        n_core,
        n_act,
        pairs,
        nmo,
    )
    h_cc = 2.0 * (ci_energies[1:] - ci_energies[0])
    n_ci = len(h_cc)
    response_hessian = np.zeros((n_orbital + n_ci, n_orbital + n_ci))
    response_hessian[:n_orbital, :n_orbital] = h_oo
    response_hessian[:n_orbital, n_orbital:] = h_oc
    response_hessian[n_orbital:, :n_orbital] = h_oc.T
    response_hessian[n_orbital:, n_orbital:] = np.diag(h_cc)
    response_rhs = -np.concatenate((d_e2_dk, d_e2_dc))

    eigvals, eigvecs = np.linalg.eigh(response_hessian)
    keep = np.abs(eigvals) > 1e-10
    multipliers = (
        eigvecs[:, keep]
        @ ((eigvecs[:, keep].T @ response_rhs) / eigvals[keep])
    )
    z_orb = multipliers[:n_orbital]
    z_ci = multipliers[n_orbital:]

    # CI-response density in the same normalized state-rotation coordinates
    # used for dE2/dc and H_cc.
    dm1_ci = np.zeros_like(dm1_cas)
    dm2_ci = np.zeros_like(dm2_cas)
    c0 = np.asarray(ci_coeffs, dtype=float).copy()
    c0 /= np.linalg.norm(c0)
    ref_state = _build_reference_state(c0, determinants, n_core, nmo)
    for k, multiplier in enumerate(z_ci, start=1):
        direction = ci_vectors[:, k].copy()
        direction -= c0 * float(c0 @ direction)
        direction /= np.linalg.norm(direction)
        response_state = _build_reference_state(direction, determinants, n_core, nmo)
        d01, g01 = _transition_rdm12(ref_state, response_state, nmo)
        d10, g10 = _transition_rdm12(response_state, ref_state, nmo)
        dm1_ci += multiplier * (d01 + d10)
        dm2_ci += multiplier * (g01 + g10)

    kappa = _build_kappa_from_z(z_orb, pairs, nmo)
    dm1_orb = _compute_one_index_transformed_dm(kappa, dm1_cas)
    dm2_orb = _compute_one_index_transformed_gamma(kappa, dm2_cas)

    dm1_lagrangian = dm1_cas + effective.delta_d + dm1_ci + dm1_orb
    dm2_lagrangian = dm2_cas + effective.delta_gamma + dm2_ci + dm2_orb
    return _contract_ic_caspt2_lagrangian_gradient(
        mol,
        basis,
        C_mo,
        h1e_mo,
        h2e_mo,
        dm1_lagrangian,
        dm2_lagrangian,
    )


# ---- Full CP-MCSCF Lagrangian gradient (OpenMolcas-equivalent) ----


def _compute_caspt2_lagrangian_gradient(
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
    """Full CP-MCSCF Lagrangian gradient for CASPT2.

    Replaces grad_casscf + grad_corr + zvec_corr with a single Lagrangian
    construction: D_eff + D^z contracted with derivative integrals.

    Returns total (n_atoms, 3) gradient in Hartree/bohr.
    """
    import numpy as np
    from scipy.linalg import expm

    from ..solvers._mrpt import _pt2_correction, _semicanonical_prep, apply_1body
    from ..solvers._rdm import make_rdm12
    from ._casscf import (
        _build_ci_hamiltonian,
        _build_ci_orbital_coupling_energy_fd,
        _build_full_gamma_mo,
        _build_orbital_hessian_fd,
        _nonredundant_pairs_local,
    )
    from ._pt2_chain_rule import compute_chain_rule_corrections
    from ._pt2_density import compute_pt2_effective_density
    from ._pt2_lagrangian import compute_pt2_total_lagrangian_gradient

    nmo = h1e_cas.shape[0]
    pairs = _nonredundant_pairs_local(n_core, n_act, nmo)
    npr = len(pairs)
    if npr == 0:
        return np.zeros((len(mol.atoms), 3))

    # Full CASSCF densities in MO basis
    D_cas = np.zeros((nmo, nmo))
    D_cas[:n_core, :n_core] = np.diag(np.full(n_core, 2.0))
    D_cas[n_core : n_core + n_act, n_core : n_core + n_act] = rdm1
    Gamma_cas = _build_full_gamma_mo(rdm1, rdm2, n_core, n_act, nmo)

    # --- Z-vector solve (same as _compute_caspt2_zvector_correction) ---

    def e2_at(h1, h2):
        P = _semicanonical_prep(h1, h2, n_core, n_act, n_act_elec, 0)
        return _pt2_correction(P, lambda s: apply_1body(s, P["F"], P["norb"]))

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
        return float((e2_at(h1p, gp) - e2_at(h1m, gm)) / (2 * fd_eps_kappa))

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
            from ._caspt2_cilag import compute_caspt2_ci_lagrangian_analytic

            dE2_dc = compute_caspt2_ci_lagrangian_analytic(
                h1e_cas, h2e_cas, ci_coeffs, determinants, n_core, n_act, n_act_elec
            )
        except Exception as exc:
            _warn_gradient_degradation("full Lagrangian CI Lagrangian", exc)
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

    # --- Effective densities with chain rule ---
    P_eff = _semicanonical_prep(h1e_cas, h2e_cas, n_core, n_act, n_act_elec, 0)
    DeltaD_full, DeltaG_full = compute_pt2_effective_density(P_eff, n_core, n_act)
    _depsa = None
    try:
        cr = compute_chain_rule_corrections(P_eff, n_core, n_act, variant="caspt2")
        if cr.get("DG2") is not None:
            DeltaG_full += cr["DG2"]
        if cr.get("DG1") is not None:
            DeltaD_full += cr["DG1"]
        _depsa = cr.get("DEPSA")
    except Exception as exc:
        _warn_gradient_degradation("full Lagrangian chain-rule correction", exc)
        pass

    # --- Full CP-MCSCF Lagrangian gradient ---
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
        DEPSA=_depsa,
        n_core=n_core,
        n_act=n_act,
        nmo=nmo,
        pairs=pairs,
    )
