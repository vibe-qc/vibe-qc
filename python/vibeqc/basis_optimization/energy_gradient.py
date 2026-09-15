"""Phase-0 analytic SCF-energy gradient w.r.t. basis parameters.

Integral-level finite-difference exponent/coefficient derivatives + the
analytic energy-weighted-density (Pulay) assembly. See
[`docs/basisset_dev/ENERGY_GRADIENT_DESIGN.md`](../../../docs/basisset_dev/ENERGY_GRADIENT_DESIGN.md).

For RHF, with density P and the energy-weighted density W = 1/2.P.F.P
(F = Hcore + G(P)), the energy gradient w.r.t. a basis parameter η is

    dE/dη = tr(P . dHcore/dη) + 1/2 tr(P . dG(P)/dη) - tr(W . dS/dη)

Each d(.)/dη is a *central finite difference of the integrals*, with P and
W frozen at the converged SCF solution. Freezing the density is exact to
first order -- the -tr(W.dS) Pulay term is precisely the wavefunction
response through the orthonormality constraint, the same reason nuclear
gradients freeze the density (cf. ``cpp/src/gradient.cpp``, where
W = 2 S_i e_i C_mui C_νi and the Pulay term contract against libint
``deriv_order=1`` integral derivatives). Here d/dR is replaced by d/dη.

The energy-weighted density is obtained without MO coefficients via the
identity W = 1/2.P.F.P: at convergence F.C = S.C.e with C S-orthonormal, so
P.F.P = (2 S_i C_iC_iᵀ) F (2 S_j C_jC_jᵀ) = 4 S_ij C_i e_j d_ij C_jᵀ
      = 2 S_i e_i C_iC_iᵀ = 2.W, hence W = 1/2.P.F.P.

This module is the **build-independent assembly**: the integral source is an
injected :class:`IntegralProvider`. The provider takes the optimiser vector
``x`` directly and maps it through the basis to the integrals, so the
returned gradient is already in optimiser space (same convention as the
finite-difference energy gradient BDIIS uses today, and directly
composable with
:func:`vibeqc.basis_optimization.gradients.penalty_gradient`):

    grad = energy_gradient_fd(provider, x) + penalty_gradient(param, x, ...)

* :class:`VibeqcIntegralProvider` -- the real provider (vibe-qc bindings);
  needs a build, verified on a compiled build.
* the Phase-1 analytic exponent-derivative integrals will drop in behind
  the same provider interface, leaving this assembly unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Protocol

import numpy as np


# --------------------------------------------------------------------------
# Pure-numpy assembly (no vibe-qc dependency)
# --------------------------------------------------------------------------


def build_j(P: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Coulomb matrix J(P)_muν = S_ls (muν|ls) P_ls.

    ``eri`` is the dense 4-index tensor in chemist notation (muν|ls) =
    ``eri[mu,ν,l,s]``. ``P`` is the density it is contracted with (total
    density for the Coulomb term in both RHF and UHF).
    """
    return np.einsum("mnls,ls->mn", eri, P, optimize=True)


def build_k(P: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Exchange matrix K(P)_muν = S_ls (mul|νs) P_ls."""
    return np.einsum("mlns,ls->mn", eri, P, optimize=True)


def build_g(P: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """RHF two-electron Fock contribution G(P) = J(P) - 1/2.K(P).

    ``P`` is the total RHF density (P = 2 S_occ C_iC_iᵀ). For the per-spin
    UHF contribution G_s = J(P_tot) - K(P_s) use :func:`build_j`/:func:`build_k`
    directly (no 1/2 -- the UHF spin density is not doubled).
    """
    return build_j(P, eri) - 0.5 * build_k(P, eri)


def fock(P: np.ndarray, hcore: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """RHF Fock F = Hcore + G(P)."""
    return hcore + build_g(P, eri)


def electronic_energy(P: np.ndarray, hcore: np.ndarray, eri: np.ndarray) -> float:
    """RHF electronic energy E = tr(P.Hcore) + 1/2 tr(P.G(P)) (excludes E_nn)."""
    return float(np.trace(P @ hcore) + 0.5 * np.trace(P @ build_g(P, eri)))


def energy_weighted_density(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Energy-weighted density W = 1/2.P.F.P (RHF).

    Equivalent to 2 S_occ e_i C_iC_iᵀ at convergence -- see module docstring.
    """
    return 0.5 * P @ F @ P


class IntegralProvider(Protocol):
    """Source of AO integrals + density as a function of the optimiser vector.

    Implementations map ``x`` -> candidate basis -> integrals, so all four
    methods accept the same optimiser-space vector. ``overlap``/``hcore``/
    ``eri`` must be smooth in ``x`` for the finite-difference derivatives to
    be meaningful; ``density`` returns the converged RHF density at ``x``.
    """

    def overlap(self, x: np.ndarray) -> np.ndarray: ...
    def hcore(self, x: np.ndarray) -> np.ndarray: ...
    def eri(self, x: np.ndarray) -> np.ndarray: ...
    def density(self, x: np.ndarray) -> np.ndarray: ...


def energy_gradient_fd(
    provider: IntegralProvider,
    x: np.ndarray,
    *,
    delta: float = 1e-4,
) -> np.ndarray:
    """RHF energy gradient dE/dx via integral-level central differences.

    Density and energy-weighted density are evaluated once at ``x`` and held
    fixed; only the integrals are differenced. Returns the gradient in the
    provider's input (optimiser) space.

    Parameters
    ----------
    provider
        Supplies overlap / hcore / eri / density as functions of ``x``.
    x
        Point at which to evaluate the gradient (optimiser space).
    delta
        Central-difference step.
    """
    x = np.asarray(x, dtype=float)
    P = np.asarray(provider.density(x), dtype=float)
    F0 = fock(P, np.asarray(provider.hcore(x)), np.asarray(provider.eri(x)))
    W = energy_weighted_density(P, F0)

    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += delta
        xm[i] -= delta
        dS = (np.asarray(provider.overlap(xp)) - np.asarray(provider.overlap(xm))) / (
            2.0 * delta
        )
        dH = (np.asarray(provider.hcore(xp)) - np.asarray(provider.hcore(xm))) / (
            2.0 * delta
        )
        dERI = (np.asarray(provider.eri(xp)) - np.asarray(provider.eri(xm))) / (
            2.0 * delta
        )
        dG = build_g(P, dERI)  # dG(P)/dη at frozen P
        grad[i] = float(
            np.trace(P @ dH) + 0.5 * np.trace(P @ dG) - np.trace(W @ dS)
        )
    return grad


# --------------------------------------------------------------------------
# Real provider -- vibe-qc bindings (needs a build; verified on a compiled build)
# --------------------------------------------------------------------------


_LTYPE_TO_L = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}


def _libint_l_sequence(atom_basis: Any) -> list[tuple[int, int]]:
    """Per-libint-shell ``(crystal_shell_idx, l)`` for one atom, in emission order.

    A crystal SP shell expands to two libint shells (s then p) that *share*
    one exponent set -- so an exponent free on an SP shell drives both, and the
    mapping returns both their positions for that crystal shell index.
    """
    seq: list[tuple[int, int]] = []
    for si, sh in enumerate(atom_basis.shells):
        if sh.shell_type == "SP":
            seq.append((si, 0))
            seq.append((si, 1))
        else:
            seq.append((si, _LTYPE_TO_L[sh.shell_type]))
    return seq


def energy_gradient_analytic(
    parametrisation: Any,
    molecule_factory: Callable[[], Any],
    library: Any,
    scf_runner: Callable[..., Any],
    x: np.ndarray,
    *,
    basis_name_prefix: str = "egrad-an",
) -> np.ndarray:
    """RHF energy gradient dE/dx from *analytic* integral exponent derivatives.

    Same Pulay assembly as :func:`energy_gradient_fd` -- density and
    energy-weighted density frozen at the converged SCF solution -- but each
    dS/dT/dV/d(muν|ls) comes from the closed-form libint exponent-derivative
    bindings (``overlap_/kinetic_/nuclear_/eri_exponent_derivative``) instead
    of finite-differencing the integrals. Returns the gradient in optimiser
    (``x``) space, drop-in interchangeable with the Phase-0 FD path.

    A free parameter names ``(element, crystal-shell, primitive)``; that maps
    to every libint shell it controls -- one per atom of the element, and both
    sides of an SP shell -- whose per-shell analytic derivatives are summed.
    The transform chain rule (dx->dη) is applied last: ``xη`` for LOG, ``x1``
    for LINEAR.

    Both ``field="exponent"`` (closed-form libint exponent derivatives) and
    ``field="coeff"`` (augmented-basis contraction-coefficient derivatives, see
    :func:`_accumulate_coeff_derivs`) are handled. SP ``coeff_s``/``coeff_p``
    are not yet analytic -- use the FD path (``energy_gradient_fd``) for those.
    """
    import vibeqc as vq  # local: vibeqc not importable without a build

    x = np.asarray(x, dtype=float)
    mol, basis, scf = _setup_scf(
        vq, parametrisation, molecule_factory, library, scf_runner, x,
        basis_name_prefix,
    )
    P = np.asarray(scf.density, dtype=float)
    F = np.asarray(scf.fock, dtype=float)
    W = energy_weighted_density(P, F)
    nbf = P.shape[0]
    shells, by_atom, atom_syms, ao_off = _shell_atom_maps(vq, basis, mol)

    grad = np.zeros(len(parametrisation.free))
    for i, spec in enumerate(parametrisation.free):
        dS, dH, dERI = _param_derivs(
            vq, basis, mol, shells, by_atom, atom_syms, ao_off,
            parametrisation, spec, nbf,
        )
        dG = build_g(P, dERI)  # dG(P)/dη at frozen P
        dE_dphys = float(
            np.trace(P @ dH) + 0.5 * np.trace(P @ dG) - np.trace(W @ dS)
        )
        grad[i] = dE_dphys * _chain_rule(spec, x[i])

    return grad


def energy_gradient_analytic_uhf(
    parametrisation: Any,
    molecule_factory: Callable[[], Any],
    library: Any,
    scf_runner: Callable[..., Any],
    x: np.ndarray,
    *,
    basis_name_prefix: str = "egrad-uhf",
) -> np.ndarray:
    """UHF (open-shell) energy gradient dE/dx from analytic integral derivatives.

    The unrestricted counterpart of :func:`energy_gradient_analytic`, using the
    spin-resolved frozen-density Pulay assembly

        dE/dη = tr(P_tot.dHcore/dη)
              + 1/2 [tr(P_a.dG_a/dη) + tr(P_b.dG_b/dη)]
              - tr((W_a + W_b).dS/dη)

    where P_tot = P_a + P_b, the per-spin two-electron contribution is
    G_s = J(P_tot) - K(P_s) (no 1/2 -- the UHF spin density is not doubled), and
    W_s = P_s.F_s.P_s is the per-spin energy-weighted density (occupation 1, so
    the RHF 1/2.P.F.P becomes P.F.P per spin). All four integral derivatives are
    spin-independent and identical to the RHF path; only the density/Fock
    contraction differs. ``scf_runner`` must return a UHF result exposing
    ``density_alpha/beta`` and ``fock_alpha/beta`` (e.g. ``vq.run_uhf``).

    Exponent and (non-SP) coeff params, same mapping and transform chain rule
    as the RHF path.
    """
    import vibeqc as vq

    x = np.asarray(x, dtype=float)
    mol, basis, scf = _setup_scf(
        vq, parametrisation, molecule_factory, library, scf_runner, x,
        basis_name_prefix,
    )
    Pa = np.asarray(scf.density_alpha, dtype=float)
    Pb = np.asarray(scf.density_beta, dtype=float)
    Fa = np.asarray(scf.fock_alpha, dtype=float)
    Fb = np.asarray(scf.fock_beta, dtype=float)
    Pt = Pa + Pb
    W = Pa @ Fa @ Pa + Pb @ Fb @ Pb  # W_a + W_b (UHF: occupation 1 per spin)
    nbf = Pt.shape[0]
    shells, by_atom, atom_syms, ao_off = _shell_atom_maps(vq, basis, mol)

    grad = np.zeros(len(parametrisation.free))
    for i, spec in enumerate(parametrisation.free):
        dS, dH, dERI = _param_derivs(
            vq, basis, mol, shells, by_atom, atom_syms, ao_off,
            parametrisation, spec, nbf,
        )
        dJ = build_j(Pt, dERI)
        dGa = dJ - build_k(Pa, dERI)  # dG_a/dη at frozen densities
        dGb = dJ - build_k(Pb, dERI)
        dE_dphys = float(
            np.trace(Pt @ dH)
            + 0.5 * (np.trace(Pa @ dGa) + np.trace(Pb @ dGb))
            - np.trace(W @ dS)
        )
        grad[i] = dE_dphys * _chain_rule(spec, x[i])

    return grad


def energy_gradient_analytic_rks(
    parametrisation: Any,
    molecule_factory: Callable[[], Any],
    library: Any,
    x: np.ndarray,
    *,
    functional: str,
    basis_name_prefix: str = "egrad-rks",
) -> np.ndarray:
    """RKS (DFT) energy gradient dE/dx from analytic integral + grid-XC terms.

    The Kohn-Sham counterpart of :func:`energy_gradient_analytic`, for a
    closed-shell RKS reference with the named ``functional``. Frozen-density
    Pulay assembly with the two-electron Coulomb (and, for hybrids, a fraction
    ``a`` of exact exchange) handled analytically via the integral derivatives,
    plus the explicit exchange-correlation term evaluated on the DFT grid:

        dE/dη = tr(P.dHcore/dη) + 1/2 tr(P.d[J - (a/2)K]/dη)
              + dE_xc/dη|_explicit - tr(W.dS/dη)

    dE_xc/dη|_explicit = S_g w_g [v_r dr/dη + 2 v_s gradr.dgradr/dη] (LDA drops the
    s term), built from the on-grid basis-function derivative dphi_mu/dη -- see
    :func:`_xc_gradient_term`. W = 1/2.P.F.P uses the converged KS Fock (which
    already contains V_xc and any hybrid exchange). The grid matches run_rks's
    (both ``build_grid(mol, opts.grid)``), so the term is consistent with the
    SCF energy that the optimiser minimises.

    Supports LDA, GGA, and global hybrids. Meta-GGA, range-separated, and
    double-hybrid functionals raise :class:`NotImplementedError`. Exponent and
    (non-SP) coefficient parameters, same mapping/chain rule as the HF paths.
    """
    import vibeqc as vq

    func = vq.Functional(functional, 1)
    if getattr(func, "is_double_hybrid", False) or getattr(func, "is_range_separated", False):
        raise NotImplementedError(
            f"analytic RKS basis gradient: {functional!r} is double-hybrid / "
            f"range-separated; not supported (use the FD path)."
        )
    if str(getattr(func, "kind", "")) not in ("XCKind.LDA", "XCKind.GGA"):
        raise NotImplementedError(
            f"analytic RKS basis gradient supports LDA/GGA(+hybrid) only; "
            f"{functional!r} kind={func.kind} (e.g. meta-GGA) not supported."
        )
    is_gga = str(func.kind) == "XCKind.GGA"
    a_hf = float(getattr(func, "hf_exchange_fraction", 0.0))

    opts = vq.RKSOptions()
    opts.functional = functional

    def _runner(mol, basis):
        return vq.run_rks(mol, basis, opts)

    x = np.asarray(x, dtype=float)
    atoms = parametrisation.unpack(x)            # current basis at x (NOT the reference)
    mol = molecule_factory()
    name = library.write_g94(atoms, basis_name=f"{basis_name_prefix}")
    basis = vq.BasisSet(mol, name)
    scf = _runner(mol, basis)
    P = np.asarray(scf.density, dtype=float)
    F = np.asarray(scf.fock, dtype=float)
    W = energy_weighted_density(P, F)
    nbf = P.shape[0]
    shells, by_atom, atom_syms, ao_off = _shell_atom_maps(vq, basis, mol)

    # Frozen XC potential on the SAME grid run_rks used (build_grid(mol, opts.grid)).
    grid = vq.build_grid(mol, opts.grid)
    pts = np.asarray(grid.points, dtype=float)
    wts = np.asarray(grid.weights, dtype=float)
    if is_gga:
        vals, gx, gy, gz = vq.evaluate_ao_with_gradient(basis, pts)
        chi = np.asarray(vals)
        gchi = np.stack([np.asarray(gx), np.asarray(gy), np.asarray(gz)], 0)
        rho = np.einsum("gm,mn,gn->g", chi, P, chi, optimize=True)
        grho = 2.0 * np.einsum("cgm,mn,gn->cg", gchi, P, chi, optimize=True)
        sigma = np.einsum("cg,cg->g", grho, grho, optimize=True)
        _, v_rho, v_sigma = func.eval_unpolarised(rho, sigma)
        v_rho = np.asarray(v_rho); v_sigma = np.asarray(v_sigma)
    else:
        chi = np.asarray(vq.evaluate_ao(basis, pts))
        gchi = grho = v_sigma = None
        rho = np.einsum("gm,mn,gn->g", chi, P, chi, optimize=True)
        _, v_rho, _ = func.eval_unpolarised(rho, np.zeros_like(rho))
        v_rho = np.asarray(v_rho)

    grid_data = (pts, wts, chi, gchi, rho, grho, v_rho, v_sigma, is_gga)

    grad = np.zeros(len(parametrisation.free))
    for i, spec in enumerate(parametrisation.free):
        targets = _spec_target_shells(spec, parametrisation, shells, by_atom, atom_syms)
        dS, dH, dERI = _param_derivs(
            vq, basis, mol, shells, by_atom, atom_syms, ao_off,
            parametrisation, spec, nbf,
        )
        dG = build_j(P, dERI)              # Coulomb; + hybrid exact exchange
        if a_hf != 0.0:
            dG = dG - 0.5 * a_hf * build_k(P, dERI)
        dExc = _xc_gradient_term(
            vq, mol, shells, ao_off, P, atoms, spec, targets, nbf, grid_data,
        )
        dE_dphys = float(
            np.trace(P @ dH) + 0.5 * np.trace(P @ dG) - np.trace(W @ dS)
        ) + dExc
        grad[i] = dE_dphys * _chain_rule(spec, x[i])

    return grad


def energy_gradient_analytic_uks(
    parametrisation: Any,
    molecule_factory: Callable[[], Any],
    library: Any,
    x: np.ndarray,
    *,
    functional: str,
    basis_name_prefix: str = "egrad-uks",
) -> np.ndarray:
    """UKS (open-shell DFT) energy gradient -- the spin-polarised counterpart of
    :func:`energy_gradient_analytic_rks`.

    Combines the UHF per-spin frozen-density Pulay assembly with the polarised
    grid exchange-correlation term:

        dE/dη = tr(P_tot.dHcore/dη)
              + 1/2 [tr(P_a.dG_a/dη) + tr(P_b.dG_b/dη)]
              + dE_xc/dη|_explicit - tr((W_a + W_b).dS/dη)

    with G_s = J(P_tot) - a.K(P_s) (``a`` = hybrid exact-exchange fraction),
    W_s = P_s.F_s.P_s (F_s the converged UKS Fock), and the spin-polarised
    explicit XC term of :func:`_xc_gradient_term_uks`. Reaches open-shell atoms
    (C, N, O, ...) at DFT level -- what the pob recipe needs. Same functional
    support, mapping and chain rule as the RKS path; reuses the (spin-
    independent) dphi_mu/dη machinery. ``scf_runner`` is ``vq.run_uks``.
    """
    import vibeqc as vq

    func = vq.Functional(functional, 2)
    if getattr(func, "is_double_hybrid", False) or getattr(func, "is_range_separated", False):
        raise NotImplementedError(
            f"analytic UKS basis gradient: {functional!r} is double-hybrid / "
            f"range-separated; not supported (use the FD path)."
        )
    if str(getattr(func, "kind", "")) not in ("XCKind.LDA", "XCKind.GGA"):
        raise NotImplementedError(
            f"analytic UKS basis gradient supports LDA/GGA(+hybrid) only; "
            f"{functional!r} kind={func.kind} (e.g. meta-GGA) not supported."
        )
    is_gga = str(func.kind) == "XCKind.GGA"
    a_hf = float(getattr(func, "hf_exchange_fraction", 0.0))

    opts = vq.UKSOptions()
    opts.functional = functional

    def _runner(mol, basis):
        return vq.run_uks(mol, basis, opts)

    x = np.asarray(x, dtype=float)
    atoms = parametrisation.unpack(x)
    mol = molecule_factory()
    name = library.write_g94(atoms, basis_name=f"{basis_name_prefix}")
    basis = vq.BasisSet(mol, name)
    scf = _runner(mol, basis)
    Pa = np.asarray(scf.density_alpha, dtype=float)
    Pb = np.asarray(scf.density_beta, dtype=float)
    Fa = np.asarray(scf.fock_alpha, dtype=float)
    Fb = np.asarray(scf.fock_beta, dtype=float)
    Pt = Pa + Pb
    W = Pa @ Fa @ Pa + Pb @ Fb @ Pb
    nbf = Pt.shape[0]
    shells, by_atom, atom_syms, ao_off = _shell_atom_maps(vq, basis, mol)

    # Frozen per-spin XC potential on the grid (build_grid(mol, opts.grid)).
    grid = vq.build_grid(mol, opts.grid)
    pts = np.asarray(grid.points, dtype=float)
    wts = np.asarray(grid.weights, dtype=float)
    if is_gga:
        vals, gx, gy, gz = vq.evaluate_ao_with_gradient(basis, pts)
        chi = np.asarray(vals)
        gchi = np.stack([np.asarray(gx), np.asarray(gy), np.asarray(gz)], 0)
        rho_a = np.einsum("gm,mn,gn->g", chi, Pa, chi, optimize=True)
        rho_b = np.einsum("gm,mn,gn->g", chi, Pb, chi, optimize=True)
        gr_a = 2.0 * np.einsum("cgm,mn,gn->cg", gchi, Pa, chi, optimize=True)
        gr_b = 2.0 * np.einsum("cgm,mn,gn->cg", gchi, Pb, chi, optimize=True)
        s_aa = np.einsum("cg,cg->g", gr_a, gr_a, optimize=True)
        s_ab = np.einsum("cg,cg->g", gr_a, gr_b, optimize=True)
        s_bb = np.einsum("cg,cg->g", gr_b, gr_b, optimize=True)
        _, v_ra, v_rb, v_saa, v_sab, v_sbb = [
            np.asarray(z) for z in func.eval_polarised(rho_a, rho_b, s_aa, s_ab, s_bb)
        ]
    else:
        chi = np.asarray(vq.evaluate_ao(basis, pts))
        gchi = gr_a = gr_b = v_saa = v_sab = v_sbb = None
        rho_a = np.einsum("gm,mn,gn->g", chi, Pa, chi, optimize=True)
        rho_b = np.einsum("gm,mn,gn->g", chi, Pb, chi, optimize=True)
        z = np.zeros_like(rho_a)
        _, v_ra, v_rb, _, _, _ = [
            np.asarray(q) for q in func.eval_polarised(rho_a, rho_b, z, z, z)
        ]

    grid_data = (pts, wts, chi, gchi, gr_a, gr_b,
                 v_ra, v_rb, v_saa, v_sab, v_sbb, is_gga)

    grad = np.zeros(len(parametrisation.free))
    for i, spec in enumerate(parametrisation.free):
        targets = _spec_target_shells(spec, parametrisation, shells, by_atom, atom_syms)
        dS, dH, dERI = _param_derivs(
            vq, basis, mol, shells, by_atom, atom_syms, ao_off,
            parametrisation, spec, nbf,
        )
        dJ = build_j(Pt, dERI)
        dGa = dJ - a_hf * build_k(Pa, dERI)
        dGb = dJ - a_hf * build_k(Pb, dERI)
        dExc = _xc_gradient_term_uks(
            vq, mol, shells, ao_off, Pa, Pb, atoms, spec, targets, nbf, grid_data,
        )
        dE_dphys = float(
            np.trace(Pt @ dH)
            + 0.5 * (np.trace(Pa @ dGa) + np.trace(Pb @ dGb))
            - np.trace(W @ dS)
        ) + dExc
        grad[i] = dE_dphys * _chain_rule(spec, x[i])

    return grad


# ---- shared assembly helpers (RHF + UHF) -----------------------------------


def _setup_scf(vq, parametrisation, molecule_factory, library, scf_runner, x,
               basis_name_prefix):
    """Unpack x -> write basis -> build molecule/basis -> run SCF. Returns
    ``(mol, basis, scf_result)``."""
    atoms = parametrisation.unpack(np.asarray(x, dtype=float))
    name = library.write_g94(atoms, basis_name=f"{basis_name_prefix}")
    mol = molecule_factory()
    basis = vq.BasisSet(mol, name)
    return mol, basis, scf_runner(mol, basis)


def _nao(sh) -> int:
    """AOs in a libint shell: 2L+1 (spherical) or (L+1)(L+2)/2 (cartesian)."""
    return 2 * sh.l + 1 if sh.pure else (sh.l + 1) * (sh.l + 2) // 2


def _shell_atom_maps(vq, basis, mol):
    """``(shells, by_atom, atom_syms, ao_off)``: the libint shell list,
    atom_index -> [global libint shell indices], per-atom element symbol, and
    the AO offset of each libint shell."""
    from ..basis_crystal import _ELEMENT_SYMBOLS

    shells = list(basis.shells())
    by_atom: dict[int, list[int]] = {}
    ao_off: list[int] = []
    off = 0
    for gi, sh in enumerate(shells):
        by_atom.setdefault(sh.atom_index, []).append(gi)
        ao_off.append(off)
        off += _nao(sh)
    atom_syms = [_ELEMENT_SYMBOLS[a.Z] for a in mol.atoms]
    return shells, by_atom, atom_syms, ao_off


def _param_derivs(vq, basis, mol, shells, by_atom, atom_syms, ao_off,
                  parametrisation, spec, nbf):
    """``(dS, dH, dERI)`` = d(integrals)/d(physical param) for one free spec.

    Exponent params use the closed-form C++ exponent-derivative bindings;
    ``coeff`` params use the augmented-basis direct term scaled by the c̃->d
    chain (``c̃_q/d_q``). Either way the returned derivatives are dI/d(physical
    param), so the downstream Pulay assembly + transform chain rule are
    identical for both. SP ``coeff_s``/``coeff_p`` are not yet supported
    analytically (raise -- use the FD path)."""
    if spec.field == "exponent":
        targets = _spec_target_shells(spec, parametrisation, shells, by_atom, atom_syms)
        return _accumulate_integral_derivs(vq, basis, mol, targets, spec.prim_idx, nbf)
    if spec.field == "coeff":
        targets = _spec_target_shells(spec, parametrisation, shells, by_atom, atom_syms)
        return _accumulate_coeff_derivs(
            vq, mol, shells, ao_off, parametrisation, spec, targets, nbf
        )
    raise NotImplementedError(
        f"{spec.display()}: analytic gradient supports exponent and (non-SP) "
        f"coeff params; field={spec.field!r}. Use energy_gradient_fd for "
        f"SP coeff_s/coeff_p."
    )


def _spec_target_shells(spec, parametrisation, shells, by_atom, atom_syms):
    """Global libint shell indices an exponent free parameter drives -- one per
    atom of its element, both sides of an SP shell -- to be summed over."""
    seq = _libint_l_sequence(parametrisation.atoms[spec.symbol])
    targets: list[int] = []
    for ai, sym in enumerate(atom_syms):
        if sym != spec.symbol:
            continue
        globals_for_atom = by_atom[ai]
        if len(globals_for_atom) != len(seq):
            raise RuntimeError(
                f"shell-count mismatch on atom {ai} ({sym}): basis has "
                f"{len(globals_for_atom)} libint shells, parametrisation "
                f"expands to {len(seq)}"
            )
        for pos, (cshell, l_expected) in enumerate(seq):
            gi = globals_for_atom[pos]
            if shells[gi].l != l_expected:
                raise RuntimeError(
                    f"shell-l mismatch on atom {ai} pos {pos}: libint l="
                    f"{shells[gi].l}, expected {l_expected}"
                )
            if cshell == spec.shell_idx:
                targets.append(gi)
    return targets


def _accumulate_integral_derivs(vq, basis, mol, targets, prim_idx, nbf):
    """Sum dS, dHcore (=dT+dV) and dERI over the target libint shells."""
    dS = np.zeros((nbf, nbf))
    dH = np.zeros((nbf, nbf))
    dERI = np.zeros((nbf, nbf, nbf, nbf))
    for gi in targets:
        dS += np.asarray(vq.overlap_exponent_derivative(basis, gi, prim_idx))
        dH += np.asarray(vq.kinetic_exponent_derivative(basis, gi, prim_idx))
        dH += np.asarray(vq.nuclear_exponent_derivative(basis, mol, gi, prim_idx))
        dERI += np.asarray(
            vq.eri_exponent_derivative(basis, gi, prim_idx)
        ).reshape(nbf, nbf, nbf, nbf)
    return dS, dH, dERI


def _accumulate_coeff_derivs(vq, mol, shells, ao_off, parametrisation, spec,
                             targets, nbf):
    """``(dS, dH, dERI)`` = dI/dd_q for a contraction-coefficient parameter.

    A contracted integral is linear in its (primitive-normalised) coefficients,
    and the SCF energy is invariant to the overall contracted normalisation, so
    only the *direct* term survives in the energy gradient:

        dI_muν/dd_q = (c̃_q/d_q) . <G_q|Ô|phi_ν>   (+ symmetric, mumu doubled)

    where G_q is the raw primitive q of shell mu and c̃_q its libint coefficient.
    <G_q|Ô|phi_ν> is obtained as the rows of an *augmented* basis -- the full basis
    plus one extra shell that is primitive q alone (unit coefficient,
    ``coefficients_pre_normalized=True``) -- so no new integral kernel is needed.
    Summed over every libint shell the parameter drives (one per atom of the
    element). ``c̃_q/d_q`` is the frozen-renormalisation dc̃_q/dd_q chain factor.
    """
    cshell = parametrisation.atoms[spec.symbol].shells[spec.shell_idx]
    d_q = float(cshell.coefficients[spec.prim_idx])
    dS = np.zeros((nbf, nbf))
    dH = np.zeros((nbf, nbf))
    dERI = np.zeros((nbf, nbf, nbf, nbf))
    chain_cd = 1.0
    for gi in targets:
        mu = shells[gi]
        chain_cd = float(mu.coefficients[spec.prim_idx]) / d_q  # c̃_q / d_q
        naux = _nao(mu)
        off = ao_off[gi]
        aux = vq.ShellInfo(mu.atom_index, mu.l, mu.pure,
                           [mu.exponents[spec.prim_idx]], [1.0], mu.origin)
        b2 = vq.BasisSet(mol, shells + [aux], "augcoeff", True)

        def _scatter2(m2):
            d = np.asarray(m2)[nbf:nbf + naux, :nbf]  # aux rows x original cols
            out = np.zeros((nbf, nbf))
            out[off:off + naux, :] += d
            out[:, off:off + naux] += d.T
            return out

        dS += _scatter2(vq.compute_overlap(b2))
        dH += _scatter2(vq.compute_kinetic(b2))
        dH += _scatter2(vq.compute_nuclear(b2, mol))

        n2 = nbf + naux
        e2 = np.asarray(vq.compute_eri(b2)).reshape(n2, n2, n2, n2)
        d4 = e2[nbf:nbf + naux, :nbf, :nbf, :nbf]  # aux in position 1
        m1 = np.zeros((nbf, nbf, nbf, nbf))
        m1[off:off + naux, :, :, :] += d4
        dERI += (m1 + m1.transpose(1, 0, 2, 3)
                 + m1.transpose(2, 3, 0, 1) + m1.transpose(3, 2, 1, 0))

    return chain_cd * dS, chain_cd * dH, chain_cd * dERI


def _chain_rule(spec, xi: float) -> float:
    """Transform chain factor da/dx: a for LOG (a = exp(x)), 1 for LINEAR."""
    phys = spec.transform.from_optim(float(xi))
    return phys if getattr(spec.transform, "name", "") == "LOG" else 1.0


def _dln_ctilde_dexponent(vq, atoms, spec):
    """dln(c̃_q)/da -- the libint contracted-coefficient response to the exponent.

    c̃_q = N_c.N_q.d_q, so this carries both the contracted (N_c) and primitive
    (N_q) normalisation responses (the caller separates them). Computed by a
    central difference of the per-shell normalisation, building the probe shell
    directly from the raw coefficients via :class:`ShellInfo`
    (``coefficients_pre_normalized=False`` -> libint embeds the normalisation) --
    *not* a ``.g94`` round-trip, whose limited text precision quantises the tiny
    exponent step and corrupts the difference. The contracted normalisation is a
    per-shell self-overlap, so a one-shell, one-atom probe basis suffices.

    ``atoms`` is the *current* (unpacked-at-x) per-element basis dict -- using the
    reference basis here would evaluate dln_ct at the wrong exponent once the
    optimiser moves x away from x0.
    """
    cs = atoms[spec.symbol].shells[spec.shell_idx]
    if cs.shell_type == "SP":
        raise NotImplementedError(
            "exponent gradient on an SP shell is not supported analytically"
        )
    l = _LTYPE_TO_L[cs.shell_type]
    pidx = spec.prim_idx
    a0 = float(cs.exponents[pidx])
    coeffs = [float(c) for c in cs.coefficients]
    h = 1e-5 * a0
    probe_mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)

    def _ctilde(da):
        exps = [float(e) for e in cs.exponents]
        exps[pidx] = a0 + da
        sh = vq.ShellInfo(0, l, True, exps, coeffs, [0.0, 0.0, 0.0])
        b = vq.BasisSet(probe_mol, [sh], "nc-probe", False)
        return float(b.shells()[0].coefficients[pidx])

    return (math.log(_ctilde(h)) - math.log(_ctilde(-h))) / (2.0 * h)


def _param_dphi_on_grid(vq, mol, shells, ao_off, atoms, spec, gi, dln_ct,
                        pts, chi, gchi, is_gga, nbf):
    """On-grid basis-function derivative dphi_mu/dη for one target libint shell.

    Returns ``(off, naux, dphi, dgphi)`` where ``dphi`` is (n_pts, naux) and
    ``dgphi`` is (3, n_pts, naux) or None (LDA). dphi_mu/dη per parameter kind:
    prim-q collocation (x c̃_q/d_q) for a coefficient; for an exponent

        dphi_mu/da = (dlnN_c/da).phi_mu + c̃_q.((2l+3)/4a - r^2).G_q

    (dlnN_c/da = ``dln_ct`` - (2l+3)/4a; the GGA gradient adds -2 c̃_q (r-A) G_q
    from grad(-r^2)). The prim-q function G_q (+gradG_q) comes from an augmented basis.
    Shared by the RKS and UKS XC-gradient terms.
    """
    mu = shells[gi]
    naux = _nao(mu)
    off = ao_off[gi]
    ctil = float(mu.coefficients[spec.prim_idx])
    aux = vq.ShellInfo(mu.atom_index, mu.l, mu.pure,
                       [mu.exponents[spec.prim_idx]], [1.0], mu.origin)
    b2 = vq.BasisSet(mol, shells + [aux], "xc-aug", True)
    if is_gga:
        av, agx, agy, agz = vq.evaluate_ao_with_gradient(b2, pts)
        aux_chi = np.asarray(av)[:, nbf:nbf + naux]
        aux_g = np.stack([np.asarray(agx), np.asarray(agy),
                          np.asarray(agz)], 0)[:, :, nbf:nbf + naux]
    else:
        aux_chi = np.asarray(vq.evaluate_ao(b2, pts))[:, nbf:nbf + naux]
        aux_g = None

    dgphi = None
    if spec.field == "exponent":
        a0 = float(mu.exponents[spec.prim_idx])
        prim = (2 * mu.l + 3) / (4.0 * a0)             # dln N_q/da
        dln_nc = dln_ct - prim                         # dln N_c/da
        r2 = np.sum((pts - np.asarray(mu.origin)) ** 2, axis=1)
        dphi = dln_nc * chi[:, off:off + naux] + (prim - r2)[:, None] * ctil * aux_chi
        if is_gga:
            rA = pts - np.asarray(mu.origin)            # (ng, 3)
            dgphi = (dln_nc * gchi[:, :, off:off + naux]
                     + (prim - r2)[None, :, None] * ctil * aux_g
                     - 2.0 * ctil * rA.T[:, :, None] * aux_chi[None, :, :])
    else:
        d_q = float(atoms[spec.symbol].shells[spec.shell_idx]
                    .coefficients[spec.prim_idx])
        chain = ctil / d_q                              # c̃_q / d_q
        dphi = chain * aux_chi
        if is_gga:
            dgphi = chain * aux_g
    return off, naux, dphi, dgphi


def _xc_gradient_term(vq, mol, shells, ao_off, P, atoms, spec, targets, nbf,
                      grid_data):
    """Explicit RKS dE_xc/dη on the DFT grid for one free parameter.

    dE_xc/dη = S_g w_g [v_r dr/dη + 2 v_s gradr.dgradr/dη] (the s term only for GGA),
    summed over the libint shells the parameter drives. dr/dη is built from the
    on-grid dphi_mu/dη (:func:`_param_dphi_on_grid`). ``atoms`` is the current
    (at-x) basis dict."""
    pts, wts, chi, gchi, rho, grho, v_rho, v_sigma, is_gga = grid_data
    dln_ct = _dln_ctilde_dexponent(vq, atoms, spec) if spec.field == "exponent" else None
    dExc = 0.0
    for gi in targets:
        off, naux, dphi, dgphi = _param_dphi_on_grid(
            vq, mol, shells, ao_off, atoms, spec, gi, dln_ct, pts, chi, gchi, is_gga, nbf
        )
        Pblk = P[off:off + naux, :]
        Pchi = chi @ Pblk.T                             # (ng, naux) = S_ν P phi_ν
        drho = 2.0 * np.einsum("ga,ga->g", dphi, Pchi, optimize=True)
        contrib = v_rho * drho
        if is_gga:
            gPchi = np.einsum("cgn,an->cga", gchi, Pblk, optimize=True)
            dgrho = 2.0 * (np.einsum("cga,ga->cg", dgphi, Pchi, optimize=True)
                           + np.einsum("ga,cga->cg", dphi, gPchi, optimize=True))
            contrib = contrib + 2.0 * v_sigma * np.einsum("cg,cg->g", grho, dgrho, optimize=True)
        dExc += float(np.sum(wts * contrib))
    return dExc


def _xc_gradient_term_uks(vq, mol, shells, ao_off, Pa, Pb, atoms, spec,
                          targets, nbf, grid_data):
    """Explicit UKS dE_xc/dη -- the spin-polarised counterpart of
    :func:`_xc_gradient_term`.

    dE_xc/dη = S_g w_g [ v_ra dr_a/dη + v_rb dr_b/dη
                        + 2 v_saa gradr_a.dgradr_a/dη + 2 v_sbb gradr_b.dgradr_b/dη
                        + v_sab (gradr_a.dgradr_b/dη + gradr_b.dgradr_a/dη) ].
    The cross s_ab term is split symmetrically across the two spins (each spin
    loop contributes v_sab . gradr_{other}.dgradr_{this}). Uses the same on-grid
    dphi_mu/dη (:func:`_param_dphi_on_grid`), contracted per spin density."""
    (pts, wts, chi, gchi, gr_a, gr_b,
     v_ra, v_rb, v_saa, v_sab, v_sbb, is_gga) = grid_data
    dln_ct = _dln_ctilde_dexponent(vq, atoms, spec) if spec.field == "exponent" else None
    # (this-spin density, v_r, gradr_this, v_s_thisthis, gradr_other, v_s_cross)
    spins = [(Pa, v_ra, gr_a, v_saa, gr_b, v_sab),
             (Pb, v_rb, gr_b, v_sbb, gr_a, v_sab)]
    dExc = 0.0
    for gi in targets:
        off, naux, dphi, dgphi = _param_dphi_on_grid(
            vq, mol, shells, ao_off, atoms, spec, gi, dln_ct, pts, chi, gchi, is_gga, nbf
        )
        for P_s, v_r, gr_s, v_ss, gr_o, v_cross in spins:
            Pblk = P_s[off:off + naux, :]
            Pchi = chi @ Pblk.T
            drho = 2.0 * np.einsum("ga,ga->g", dphi, Pchi, optimize=True)
            contrib = v_r * drho
            if is_gga:
                gPchi = np.einsum("cgn,an->cga", gchi, Pblk, optimize=True)
                dgrho = 2.0 * (np.einsum("cga,ga->cg", dgphi, Pchi, optimize=True)
                               + np.einsum("ga,cga->cg", dphi, gPchi, optimize=True))
                contrib = (contrib
                           + 2.0 * v_ss * np.einsum("cg,cg->g", gr_s, dgrho, optimize=True)
                           + v_cross * np.einsum("cg,cg->g", gr_o, dgrho, optimize=True))
            dExc += float(np.sum(wts * contrib))
    return dExc


class VibeqcIntegralProvider:
    """:class:`IntegralProvider` backed by vibe-qc's AO-integral bindings.

    For each ``x``: unpack -> write the candidate basis as a temp ``.g94``
    (via the entered :class:`TempBasisLibrary`), build ``vq.BasisSet``, and
    return

    * ``overlap`` = ``vq.compute_overlap(basis)``
    * ``hcore``   = ``vq.compute_kinetic(basis) + vq.compute_nuclear(basis, mol)``
    * ``eri``     = ``vq.compute_eri(basis)``  (dense (muν|ls))
    * ``density`` = ``scf_runner(mol, basis).density``

    Mirrors the basis-writing path of
    :class:`vibeqc.basis_optimization.objective.SinglePointEnergy`. Needs a
    built vibe-qc; the assembly above is validated build-free against a mock
    RHF, and this provider is validated on a compiled build -- see
    ``scripts/basisset_dev/verify_energy_gradient.py``.
    """

    def __init__(
        self,
        parametrisation: Any,
        molecule_factory: Callable[[], Any],
        library: Any,
        scf_runner: Callable[..., Any],
        *,
        basis_name_prefix: str = "egrad",
    ) -> None:
        self.parametrisation = parametrisation
        self.molecule_factory = molecule_factory
        self.library = library
        self.scf_runner = scf_runner
        self.basis_name_prefix = basis_name_prefix
        self._calls = 0

    def _basis(self, x: np.ndarray):
        import vibeqc as vq  # local: vibeqc not importable without a build

        atoms = self.parametrisation.unpack(np.asarray(x, dtype=float))
        self._calls += 1
        name = self.library.write_g94(
            atoms, basis_name=f"{self.basis_name_prefix}-{self._calls:06d}"
        )
        mol = self.molecule_factory()
        return vq, mol, vq.BasisSet(mol, name)

    def overlap(self, x: np.ndarray) -> np.ndarray:
        vq, _mol, basis = self._basis(x)
        return np.asarray(vq.compute_overlap(basis))

    def hcore(self, x: np.ndarray) -> np.ndarray:
        vq, mol, basis = self._basis(x)
        return np.asarray(vq.compute_kinetic(basis)) + np.asarray(
            vq.compute_nuclear(basis, mol)
        )

    def eri(self, x: np.ndarray) -> np.ndarray:
        vq, _mol, basis = self._basis(x)
        return np.asarray(vq.compute_eri(basis))

    def density(self, x: np.ndarray) -> np.ndarray:
        vq, mol, basis = self._basis(x)
        return np.asarray(self.scf_runner(mol, basis).density)
