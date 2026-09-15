"""Build-independent periodic (k-point) SCF-energy gradient w.r.t. basis params.

The periodic counterpart of
[`energy_gradient.py`](energy_gradient.py)'s ``energy_gradient_fd`` -- the
**build-independent assembly only** (Phase P0). It mirrors the molecular
frozen-density Pulay structure, summed over the Brillouin zone with complex
per-k matrices:

    dE/dη = S_k w_k . Re[ tr(P(k).dHcore(k)/dη) + 1/2 tr(P(k).dG(k)/dη)
                          - tr(W(k).dS(k)/dη) ]

with the per-k energy-weighted density W(k) = 1/2.P(k).F(k).P(k),
F(k) = Hcore(k) + G(k), and normalised k-weights (S_k w_k = 1). P(k), S(k),
Hcore(k), G(k) are Hermitian; the BZ-summed traces are real, and taking Re per
k is the physical per-k contribution (the molecular case is the single Γ-point
limit, P real-symmetric, Re a no-op).

As in the molecular Phase-0, the integral source is an injected
:class:`PeriodicIntegralProvider`. The hard periodic-specific pieces drop in
behind it (see ``docs/basisset_dev/PERIODIC_GRADIENT_DESIGN.md``):

* **Phase P1 (this module, build-dependent).** The Bloch-summed *one-electron*
  integral derivatives dS(k)/da = S_R e^{ik.R} dS(R)/da and the kinetic
  analogue dT(k)/da -- :func:`bloch_summed_one_electron_exponent_derivatives`.
  These are clean two-centre lattice sums, built by reusing the molecular
  exponent-derivative bindings (``overlap_/kinetic_exponent_derivative``)
  between a home-cell shell and its image at a lattice vector R. Validated
  against a central FD of the Bloch-summed lattice integral at fixed k in
  ``tests/basisset_dev/test_periodic_one_electron_exponent_deriv.py``.
* **Phase P2 (assembly skeleton here; derivative kernel pending).** The full
  per-k Pulay gradient is wired in :func:`periodic_energy_gradient_analytic`,
  which consumes the P1 one-electron derivatives plus an injected
  :class:`PeriodicCoulombNuclearDerivativeProvider` for the nuclear dV(k)/da and
  two-electron dG(k)/da. Those last two are coupled to the Ewald gauge of the
  periodic SCF (the GDF multi-k path builds V via an Ewald-3D-gauge nuclear
  lattice and J/K via the Ewald Fock builder), so they share machinery with
  ``periodic_gdf_gradient.py`` and are produced by the REQUIREMENTS-PERIODIC.md
  R13 lattice-summed exponent-derivative kernel (not yet landed). Until then the
  assembly is reachable only through a mock provider in tests (gated).
* **Phase P3 (RKS + UKS, LDA/GGA here).** The periodic explicit-XC term
  ``dE_xc/dη`` -- :func:`periodic_xc_param_gradient_term` (closed-shell) and
  :func:`periodic_xc_param_gradient_term_uks` (open-shell), with their
  ``build_periodic_xc_gradient_grid``/``_uks`` frozen-grid builders. Real-space
  lattice density on the periodic Becke grid, reusing the molecular
  ``_param_dphi_on_grid`` per cell; validated against a frozen-density
  ``build_xc_periodic``/``_uks`` E_xc finite difference. Meta-GGA later.

The frozen-density assembly itself (top of this module) is validated build-free
against a mock provider, exactly like
``tests/basisset_dev/test_energy_gradient_assembly.py`` does for the molecular
assembly. The per-k converged P(k)/F(k)/S(k)/Hcore(k) the assembly consumes are
already exposed by the GDF multi-k SCF result (``PeriodicKRHFGDFResult``), so no
new periodic-SCF binding is needed to feed it (the design doc's "Phase P4").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np


def energy_weighted_density_k(P_k: np.ndarray, F_k: np.ndarray) -> np.ndarray:
    """Per-k energy-weighted density W(k) = 1/2.P(k).F(k).P(k) (Hermitian)."""
    return 0.5 * P_k @ F_k @ P_k


def periodic_electronic_energy(
    densities: Sequence[np.ndarray],
    hcores: Sequence[np.ndarray],
    gmatrices: Sequence[np.ndarray],
    kweights: Sequence[float],
) -> float:
    """BZ-summed electronic energy S_k w_k Re[tr(P.Hcore) + 1/2 tr(P.G)] (no E_nn).

    ``gmatrices[k]`` is the two-electron Fock contribution G(k) at ``densities``
    (the periodic analogue of the molecular ``build_g``; for GDF/Ewald it
    couples all k and is supplied by the provider rather than rebuilt here).
    """
    e = 0.0
    for P, h, g, w in zip(densities, hcores, gmatrices, kweights):
        e += w * float(np.real(np.trace(P @ h) + 0.5 * np.trace(P @ g)))
    return e


class PeriodicIntegralProvider(Protocol):
    """Source of per-k AO integrals + density as a function of the optimiser x.

    Every method returns one (n_bf x n_bf) **complex** matrix per k-point, in
    the order of :attr:`kweights`. ``overlap``/``hcore``/``gmatrix`` must be
    smooth in ``x`` for the finite-difference derivatives to be meaningful;
    ``gmatrix`` builds the two-electron Fock contribution G(k) at the *frozen*
    converged density (so its x-dependence is through the integrals only, like
    the molecular ``build_g(P_frozen, eri(x))``). ``density`` returns the
    converged per-k density P(k) at ``x``.
    """

    kweights: Sequence[float]

    def overlap(self, x: np.ndarray) -> Sequence[np.ndarray]: ...
    def hcore(self, x: np.ndarray) -> Sequence[np.ndarray]: ...
    def gmatrix(self, x: np.ndarray) -> Sequence[np.ndarray]: ...
    def density(self, x: np.ndarray) -> Sequence[np.ndarray]: ...


def periodic_energy_gradient_fd(
    provider: PeriodicIntegralProvider,
    x: np.ndarray,
    *,
    delta: float = 1e-4,
) -> np.ndarray:
    """Periodic RHF/RKS energy gradient dE/dx via per-k integral central diffs.

    Per-k density and energy-weighted density are evaluated once at ``x`` and
    held fixed; only the integrals are differenced. Returns the gradient in the
    provider's optimiser space. The single-k (Γ-only, real) limit reduces to the
    molecular :func:`energy_gradient_fd`.
    """
    x = np.asarray(x, dtype=float)
    kw = np.asarray(provider.kweights, dtype=float)
    P = [np.asarray(Pk) for Pk in provider.density(x)]
    H0 = [np.asarray(Hk) for Hk in provider.hcore(x)]
    G0 = [np.asarray(Gk) for Gk in provider.gmatrix(x)]
    W = [energy_weighted_density_k(Pk, Hk + Gk) for Pk, Hk, Gk in zip(P, H0, G0)]

    grad = np.zeros(len(x))
    for i in range(len(x)):
        xp = x.copy(); xp[i] += delta
        xm = x.copy(); xm[i] -= delta
        Sp, Sm = provider.overlap(xp), provider.overlap(xm)
        Hp, Hm = provider.hcore(xp), provider.hcore(xm)
        Gp, Gm = provider.gmatrix(xp), provider.gmatrix(xm)
        gi = 0.0
        for k in range(len(kw)):
            dS = (np.asarray(Sp[k]) - np.asarray(Sm[k])) / (2.0 * delta)
            dH = (np.asarray(Hp[k]) - np.asarray(Hm[k])) / (2.0 * delta)
            dG = (np.asarray(Gp[k]) - np.asarray(Gm[k])) / (2.0 * delta)
            gi += kw[k] * float(np.real(
                np.trace(P[k] @ dH) + 0.5 * np.trace(P[k] @ dG)
                - np.trace(W[k] @ dS)
            ))
        grad[i] = gi
    return grad


# --------------------------------------------------------------------------
# Phase P1 -- Bloch-summed one-electron exponent derivatives (needs a build)
# --------------------------------------------------------------------------
#
# The periodic one-electron matrix at k is the Bloch sum of its real-space
# lattice blocks (textbook tight-binding identity, the same convention as the
# native ``bloch_sum`` kernel, ``M(k) = S_g e^{+i k.R_g} M(R_g)``):
#
#     S(k)_muν = S_R e^{+i k.R} <phi_mu(r) | phi_ν(r - R)>
#
# so its derivative w.r.t. a primitive Gaussian exponent a is the Bloch sum of
# the per-cell real-space derivatives,
#
#     dS(k)/da = S_R e^{+i k.R} dS(R)/da ,   S(R)_muν = <phi_mu(0) | phi_ν(R)> ,
#
# and likewise for the kinetic matrix T. Each real-space block derivative
# dS(R)/da is a *molecular* two-centre integral derivative between a home-cell
# shell and the same shell translated to the image cell at Cartesian lattice
# vector R -- exactly the closed-form exponent derivative already validated for
# the molecular gradient (``cpp/src/basis_param_gradient.cpp``, bound as
# ``overlap_/kinetic_exponent_derivative``). Because the *same* exponent a lives
# on both the home shell and its image, dS(R)/da is the sum of the derivative
# w.r.t. the home copy of the shell (a in the bra) and w.r.t. the image copy
# (a in the ket); the molecular binding returns the full matrix derivative
# w.r.t. one shell's exponent (both roles), so summing the two copies and
# reading the homeximage cross-block gives dS(R)/da.
#
# Nuclear attraction dV(k)/da and the two-electron dG(k)/da are Phase P2 -- they
# carry the Ewald gauge and are not clean two-centre sums (see module docstring).


def _image_shell_infos(vq, home_shells, n_home_atoms, r_cart):
    """``home_shells`` translated to the image cell at Cartesian ``r_cart``.

    Each image shell is a fresh :class:`ShellInfo` with the same l / purity /
    exponents / (libint-normalised) coefficients, its origin shifted by
    ``r_cart``, and its ``atom_index`` offset past the home atoms so it binds to
    the image atom in the doubled molecule. Coefficients are passed through
    verbatim (``coefficients_pre_normalized=True`` on the rebuilt basis), so the
    home+image basis reproduces the SCF's real-space lattice blocks exactly.
    """
    out = []
    for s in home_shells:
        out.append(
            vq.ShellInfo(
                s.atom_index + n_home_atoms,
                s.l,
                s.pure,
                [float(e) for e in s.exponents],
                [float(c) for c in s.coefficients],
                [s.origin[0] + r_cart[0], s.origin[1] + r_cart[1], s.origin[2] + r_cart[2]],
            )
        )
    return out


def bloch_summed_one_electron_exponent_derivatives(
    system: Any,
    basis: Any,
    target_shells: Sequence[int],
    prim_idx: int,
    kpoints_cart: np.ndarray,
    *,
    lattice_opts: Any = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Bloch-summed dS(k)/da and dT(k)/da for one primitive exponent (Phase P1).

    Differentiates the periodic overlap S(k) and kinetic T(k) w.r.t. the
    ``prim_idx``-th primitive exponent a of the crystal shell instanced by the
    home libint shells ``target_shells`` (one entry per symmetry-equivalent atom
    of the element carrying the free exponent -- the molecular
    :func:`vibeqc.basis_optimization.energy_gradient._spec_target_shells`
    produces exactly this index list for a ``FreeSpec``). a drives every image of
    those shells, so the lattice sum couples all cells.

    Returns ``(dS_k, dT_k)``, each a list (one per row of ``kpoints_cart``) of
    complex ``(nbf, nbf)`` arrays holding dM(k)/da -- the derivative w.r.t. the
    *physical* exponent (no optimiser-space chain rule; the caller applies the
    LOG/LINEAR factor, matching the molecular path). The home (R = 0) block is
    the plain molecular derivative; every R != 0 block is the homeximage
    cross-block of the doubled-cell molecular derivative, summed over the home
    and image copies of each target shell. The lattice cells (and hence the k
    phases) are taken from the SCF's own overlap lattice at ``lattice_opts`` so
    the result differentiates exactly the S(k)/T(k) the SCF assembles.

    ``lattice_opts`` must match the SCF (defaults to ``LatticeSumOptions()``).
    Needs a built vibe-qc; validated against a Bloch-summed integral FD in
    ``tests/basisset_dev/test_periodic_one_electron_exponent_deriv.py``.
    """
    import vibeqc as vq  # local: vibeqc not importable without a build

    if lattice_opts is None:
        lattice_opts = vq.LatticeSumOptions()

    kpoints = np.atleast_2d(np.asarray(kpoints_cart, dtype=float))
    mol = system.unit_cell_molecule()
    n_home_atoms = len(mol.atoms)
    home_shells = list(basis.shells())
    n_home_shells = len(home_shells)
    nbf = basis.nbasis
    targets = [int(t) for t in target_shells]

    # Authoritative cell list: the SCF's own overlap lattice at these opts. Its
    # ``.cells[i].r_cart`` are the Cartesian lattice vectors R; both S(k) and
    # T(k) are summed over this same set (shared ``cutoff_bohr``).
    s_lat = vq.compute_overlap_lattice(basis, system, lattice_opts)
    cells = s_lat.cells

    home_atoms = [vq.Atom(int(a.Z), list(a.xyz)) for a in mol.atoms]

    dS_k = [np.zeros((nbf, nbf), dtype=complex) for _ in range(len(kpoints))]
    dT_k = [np.zeros((nbf, nbf), dtype=complex) for _ in range(len(kpoints))]
    if not targets:
        return dS_k, dT_k

    for cell in cells:
        r = np.asarray(cell.r_cart, dtype=float).ravel()
        if np.allclose(r, 0.0):
            dS = np.zeros((nbf, nbf))
            dT = np.zeros((nbf, nbf))
            for gi in targets:
                dS += np.asarray(vq.overlap_exponent_derivative(basis, gi, prim_idx))
                dT += np.asarray(vq.kinetic_exponent_derivative(basis, gi, prim_idx))
        else:
            image_atoms = [
                vq.Atom(int(a.Z), [a.xyz[0] + r[0], a.xyz[1] + r[1], a.xyz[2] + r[2]])
                for a in mol.atoms
            ]
            # Doubled cell is 2x the electrons of the home cell, so an even count
            # -- multiplicity 1 is always consistent (overlap/kinetic ignore Z).
            comb_mol = vq.Molecule(home_atoms + image_atoms, multiplicity=1)
            comb_shells = home_shells + _image_shell_infos(
                vq, home_shells, n_home_atoms, r
            )
            comb = vq.BasisSet(comb_mol, comb_shells, "periodic-2cell", True)
            accS = np.zeros((2 * nbf, 2 * nbf))
            accT = np.zeros((2 * nbf, 2 * nbf))
            for gi in targets:
                for g in (gi, gi + n_home_shells):  # home copy (bra) + image copy (ket)
                    accS += np.asarray(vq.overlap_exponent_derivative(comb, g, prim_idx))
                    accT += np.asarray(vq.kinetic_exponent_derivative(comb, g, prim_idx))
            dS = accS[0:nbf, nbf : 2 * nbf]  # home rows x image cols = dS(R)/da
            dT = accT[0:nbf, nbf : 2 * nbf]
        for ik, k in enumerate(kpoints):
            phase = np.exp(1j * float(np.dot(k, r)))
            dS_k[ik] += phase * dS
            dT_k[ik] += phase * dT
    return dS_k, dT_k


# --------------------------------------------------------------------------
# Phase P2 (skeleton) -- full periodic analytic gradient assembly
# --------------------------------------------------------------------------
#
# The complete per-k Pulay gradient (module top) is
#
#     dE/dη = S_k w_k Re[ tr(P(k).dHcore(k)/dη) + 1/2 tr(P(k).dG(k)/dη)
#                         - tr(W(k).dS(k)/dη) ],  W(k) = 1/2 P(k) F(k) P(k).
#
# With dHcore = dT + dV, Phase P1 supplies the clean two-centre dS(k) and dT(k)
# (``bloch_summed_one_electron_exponent_derivatives``). The Ewald-gauge nuclear
# dV(k) and two-electron dG(k) are Phase P2: they need the lattice-summed
# exponent-derivative 3c/2c kernel asked for in REQUIREMENTS-PERIODIC.md R13.
#
# This section is the *wiring* that consumes those pieces. The derivative source
# for dV(k)/dG(k) is an injected :class:`PeriodicCoulombNuclearDerivativeProvider`
# -- exactly as the molecular Phase-0 assembly injected ``IntegralProvider`` and
# the Phase-1 analytic integrals dropped in behind it. Until R13 lands there is
# no production provider, so the full gradient is reachable only through a mock
# provider in tests; the assembly itself (and its P1 one-electron half) is
# validated now, so the day R13 ships the gradient is a drop-in.
#
# Scope of the skeleton: closed-shell periodic RHF (and the exact-exchange part
# of a global hybrid, carried inside dG). The explicit grid-XC term dE_xc/dη for
# pure-DFT crystals is Phase P3 (an additive ``+ dExc`` not assembled here), and
# UHF/UKS spin resolution is later still. Exponent parameters only (periodic
# coefficient derivatives are future work); coeff specs raise NotImplementedError.


def _periodic_pulay_contract(densities, focks, kweights, dS_k, dHcore_k, dG_k):
    """One parameter's dE/dη (physical-param) from per-k integral derivatives.

    The build-independent core of the analytic assembly:

        S_k w_k Re[ tr(P(k).dHcore(k)) + 1/2 tr(P(k).dG(k)) - tr(W(k).dS(k)) ]

    with W(k) = 1/2 P(k) F(k) P(k). All ``*_k`` arguments are length-nk sequences
    of (nbf, nbf) matrices (complex per k); ``densities``/``focks`` are the
    converged P(k)/F(k). No chain rule (the caller maps dη->dx). Pure numpy, so it
    is validated build-free against a mock multi-k total-energy FD, exactly like
    :func:`periodic_energy_gradient_fd`.
    """
    g = 0.0
    for P, F, w, dS, dH, dG in zip(densities, focks, kweights, dS_k, dHcore_k, dG_k):
        P = np.asarray(P); F = np.asarray(F)
        W = energy_weighted_density_k(P, F)
        g += float(w) * float(np.real(
            np.trace(P @ np.asarray(dH))
            + 0.5 * np.trace(P @ np.asarray(dG))
            - np.trace(W @ np.asarray(dS))
        ))
    return g


class PeriodicCoulombNuclearDerivativeProvider(Protocol):
    """Source of the Ewald-gauge dV(k)/dη and two-electron dG(k)/dη (Phase P2).

    The piece P1 cannot supply: the nuclear-attraction and Coulomb/exchange
    basis-parameter derivatives, which are not clean two-centre sums (they carry
    the periodic Ewald gauge). Implemented by the REQUIREMENTS-PERIODIC.md R13
    lattice-summed exponent-derivative 3c/2c kernel once it lands; a mock/FD
    implementation stands in for tests until then.

    Both methods take the home libint-shell indices ``target_shells`` instancing
    the crystal shell whose exponent varies, and the primitive index, and return
    one (nbf, nbf) complex matrix per k-point (the provider holds the per-k
    converged density it differentiates ``G`` at, frozen, like the molecular
    ``build_g(P_frozen, dERI)``)."""

    def nuclear_kpoint_derivative(
        self, target_shells: Sequence[int], prim_idx: int
    ) -> Sequence[np.ndarray]:
        """dV(k)/da per k-point (Ewald-gauge nuclear attraction)."""
        ...

    def gmatrix_kpoint_derivative(
        self, target_shells: Sequence[int], prim_idx: int
    ) -> Sequence[np.ndarray]:
        """dG(k)/da per k-point at the frozen converged density."""
        ...


def periodic_energy_gradient_analytic(
    parametrisation: Any,
    system: Any,
    basis: Any,
    scf_result: Any,
    coulomb_nuclear_provider: PeriodicCoulombNuclearDerivativeProvider,
    x: np.ndarray,
    *,
    lattice_opts: Any = None,
) -> np.ndarray:
    """Full periodic RHF analytic basis-parameter gradient (Phase P2 skeleton).

    Assembles ``dE/dx`` in optimiser space from the frozen-density per-k Pulay
    expression, drawing dS(k)/dT(k) from Phase P1
    (:func:`bloch_summed_one_electron_exponent_derivatives`) and dV(k)/dG(k) from
    the injected ``coulomb_nuclear_provider`` (the R13 kernel). The converged
    per-k density P(k), Fock F(k), k-points and k-weights are read from a GDF
    multi-k SCF result (``PeriodicKRHFGDFResult`` exposes ``density``, ``fock``,
    ``kpoints_cart``, ``kpoint_weights`` per k -- design-doc "P4", already
    satisfied). The k-weights are used as the SCF stored them, so the gradient is
    consistent with the energy the SCF minimised.

    Per free spec the physical-param derivative is contracted by
    :func:`_periodic_pulay_contract` and then mapped to optimiser space by the
    same LOG/LINEAR chain rule as the molecular path. Exponent params only;
    ``coeff`` specs raise :class:`NotImplementedError` (periodic coefficient
    derivatives are future work), as do SP exponents.

    **Gated**: there is no production ``coulomb_nuclear_provider`` until R13
    lands, so in production this raises until one is supplied. The wiring (P1 +
    assembly + chain rule + k-weights) is exercised through a mock provider in
    ``tests/basisset_dev/test_periodic_energy_gradient_analytic_wiring.py``; the
    build-free contraction is covered by ``_periodic_pulay_contract`` tests.
    """
    import vibeqc as vq  # local: vibeqc not importable without a build
    from .energy_gradient import (  # reuse the molecular spec->shell mapping
        _shell_atom_maps,
        _spec_target_shells,
        _chain_rule,
    )

    if coulomb_nuclear_provider is None:
        raise NotImplementedError(
            "periodic_energy_gradient_analytic needs a "
            "PeriodicCoulombNuclearDerivativeProvider for dV(k)/dG(k); the "
            "production kernel is REQUIREMENTS-PERIODIC.md R13 (not yet landed). "
            "Until then, only the Phase-P1 one-electron derivatives "
            "(bloch_summed_one_electron_exponent_derivatives) are available, or "
            "supply a mock/FD provider."
        )
    if lattice_opts is None:
        lattice_opts = vq.LatticeSumOptions()

    x = np.asarray(x, dtype=float)
    densities = [np.asarray(P) for P in scf_result.density]
    focks = [np.asarray(F) for F in scf_result.fock]
    kweights = np.asarray(scf_result.kpoint_weights, dtype=float)
    kpoints_cart = np.atleast_2d(np.asarray(scf_result.kpoints_cart, dtype=float))

    mol = system.unit_cell_molecule()
    shells, by_atom, atom_syms, _ao_off = _shell_atom_maps(vq, basis, mol)

    grad = np.zeros(len(parametrisation.free))
    for i, spec in enumerate(parametrisation.free):
        if spec.field != "exponent":
            raise NotImplementedError(
                f"{spec.display()}: periodic analytic gradient currently supports "
                f"exponent params only (field={spec.field!r}); periodic coefficient "
                f"derivatives are future work."
            )
        targets = _spec_target_shells(spec, parametrisation, shells, by_atom, atom_syms)
        dS_k, dT_k = bloch_summed_one_electron_exponent_derivatives(
            system, basis, targets, spec.prim_idx, kpoints_cart,
            lattice_opts=lattice_opts,
        )
        dV_k = [np.asarray(m) for m in
                coulomb_nuclear_provider.nuclear_kpoint_derivative(targets, spec.prim_idx)]
        dG_k = [np.asarray(m) for m in
                coulomb_nuclear_provider.gmatrix_kpoint_derivative(targets, spec.prim_idx)]
        dHcore_k = [dT + dV for dT, dV in zip(dT_k, dV_k)]
        dE_dphys = _periodic_pulay_contract(
            densities, focks, kweights, dS_k, dHcore_k, dG_k
        )
        grad[i] = dE_dphys * _chain_rule(spec, x[i])
    return grad


# --------------------------------------------------------------------------
# Phase P3 -- periodic explicit exchange-correlation gradient term (RKS)
# --------------------------------------------------------------------------
#
# A KS reference adds an explicit XC term to the assembly:
#
#     dE_xc/dη = S_g w_g [ v_r(g) dr/dη(g) + 2 v_s(g) gradr(g).dgradr/dη(g) ]
#
# (the s term is GGA-only), with the *physical periodic* density in real-space
# lattice form -- the full cross-cell sum that ``build_xc_periodic`` assembles
# (cpp/src/periodic_xc.cpp, since the 295f7ada dense-crystal fix):
#
#     r(r) = S_a S_s S_muν chi_mu(r - R_a) P_muν(s - a) chi_ν(r - R_s)
#
# on the periodic Becke grid. BOTH AOs range over lattice cells: a bra cell a and
# a ket cell s, with the real-space density block taken at their difference
# g = s - a. chi(r - R_c) is the home AO evaluated at the shifted points (r - R_c).
# (The earlier builder summed only the home-bra slice r = S_h chi_0 P(h) chi_h,
# which differs from the cross-cell energy the SCF minimises by ~0.09 Ha on the
# dense 1-D test chains -- a latent gradient bug fixed here.) Ket cells are
# screened by ``cutoff_bohr`` and bra-image cells by
# ``min(becke_image_radius_bohr, cutoff_bohr)``, exactly mirroring the C++
# ``active_density_cells`` split.
#
# dchi_mu/dη is the *molecular* on-grid derivative ``_param_dphi_on_grid`` (reused
# verbatim), evaluated at the shifted points (r - R_c) of whichever cell the AO
# sits in. The target shell now appears in BOTH the bra-image chi_mu(r - R_a) AND
# the ket chi_ν(r - R_s), so dr/dη carries a derivative term for each -- the
# "home + image copies of the shared parameter" structure of Phase P1, extended to
# the bra-image cells (η drives the target shell whichever cell role it plays).
# v_r/v_s are frozen at the converged density (libxc, the same Functional the SCF
# used).
#
# The Python density assembly reproduces ``build_xc_periodic``'s E_xc to ~1e-11
# (pinned in the convention test), so dE_xc/dη is validated against a central FD of
# E_xc at frozen density. Closed-shell RKS, LDA + GGA; for a global hybrid the
# exact exchange rides in the two-electron term (P2), not here. Meta-GGA /
# range-separated / double-hybrid raise. UKS is below.

# Fallback periodic Becke partition reach (bohr) for the cross-cell bra-image
# screen when LatticeSumOptions::becke_image_radius_bohr is unset (0). Matches
# the C++ kDefaultBeckeImageRadiusBohr (cpp/src/periodic_xc.cpp) and
# build_periodic_becke_grid's default image_radius_bohr.
_DEFAULT_BECKE_IMAGE_RADIUS_BOHR = 10.0


def _density_has_cross_cell(pblocks, cell_index, tol=1e-12):
    """True iff any non-home density block is non-negligible (mirrors the C++
    ``density_has_cross_cell_blocks``). A density with only the home (g=0) block
    populated is a molecular-limit density whose physical grid density is the
    home-cell form chi_0 P(0) chi_0 -- replicating image-bra copies would
    over-count -- so the bra stays home-only in that case."""
    for blk, idx in zip(pblocks, cell_index):
        if idx == (0, 0, 0):
            continue
        if blk.size and float(np.max(np.abs(blk))) > tol:
            return True
    return False


def _xc_cross_cell_braket(system, cell_index, cells_rcart, opts, has_cross_cell):
    """Active ket / bra-image cells + (a, s, gp) pairs for the cross-cell density.

    Mirrors the C++ ``active_density_cells`` / ``build_braket_pairs`` split in
    ``cpp/src/periodic_xc.cpp`` so the gradient builder assembles the SAME
    physical periodic density r = S_{(a,s)} chi_a P(s-a) chi_s as
    ``build_xc_periodic``. A cell is active when an image atom (home atom + R_c)
    comes within the screen radius of a home atom. Ket cells use
    ``opts.cutoff_bohr``; bra-image cells use
    ``min(becke_image_radius_bohr or default, cutoff_bohr)`` when the density
    carries cross-cell blocks (else the home cell only). Returns
    ``(active, pairs)`` with ``active`` the ket-cell positions and ``pairs`` a
    list of ``(a, s, gp)`` positions, gp resolving the block P(idx(s) - idx(a))."""
    home = np.array([list(a.xyz) for a in system.unit_cell], dtype=float)

    def _near_cells(radius):
        r2 = float(radius) * float(radius)
        out = []
        for c, (idx, R) in enumerate(zip(cell_index, cells_rcart)):
            if idx == (0, 0, 0):
                out.append(c)
                continue
            near = False
            for pa in home + R[None, :]:            # image atoms (home + R_c)
                d = home - pa[None, :]
                if float(np.min(np.einsum("ij,ij->i", d, d))) <= r2:
                    near = True
                    break
            if near:
                out.append(c)
        return out

    active = _near_cells(opts.cutoff_bohr)
    becke = float(opts.becke_image_radius_bohr)
    bra_radius = min(becke if becke > 0.0 else _DEFAULT_BECKE_IMAGE_RADIUS_BOHR,
                     float(opts.cutoff_bohr))
    if has_cross_cell:
        bra_cells = _near_cells(bra_radius)
    else:
        bra_cells = [c for c in active if cell_index[c] == (0, 0, 0)]

    pos = {cell_index[c]: c for c in range(len(cell_index))}
    pairs = []
    for a in bra_cells:
        ia = cell_index[a]
        for s in active:
            isx = cell_index[s]
            g = (isx[0] - ia[0], isx[1] - ia[1], isx[2] - ia[2])
            gp = pos.get(g)
            if gp is not None:
                pairs.append((a, s, gp))
    return active, pairs


def _periodic_grid_param_dchi(vq, mol, nbf, shells, ao_off, atoms, spec, targets,
                              dln_ct, is_gga, pts, cells, active, chih):
    """Per-cell dchi/dη on the grid for the cross-cell lattice density.

    Returns ``dchih``: a list aligned with the density cells, ``dchih[c] =
    (dch, dgh)`` for every active cell c (the on-grid parameter derivative
    (n_pts, nbf) and its gradient (3, n_pts, nbf) or None, evaluated at the
    shifted points ``pts - R_c``), and ``None`` for inactive cells. Wraps the
    molecular ``_param_dphi_on_grid`` summed over the target shells; reused by
    the RKS and UKS XC gradient terms (the basis-function derivative does not
    depend on spin). Each active cell appears as both a bra-image and a ket in
    the pair sum, so computing dchi once per active cell covers both roles."""
    from .energy_gradient import _param_dphi_on_grid

    def _dchi(p, c_ao, gc_ao):
        D = np.zeros((len(p), nbf))
        Dg = np.zeros((3, len(p), nbf)) if is_gga else None
        for gi in targets:
            off, naux, dphi, dgphi = _param_dphi_on_grid(
                vq, mol, shells, ao_off, atoms, spec, gi, dln_ct, p, c_ao, gc_ao,
                is_gga, nbf,
            )
            D[:, off:off + naux] += dphi
            if is_gga:
                Dg[:, :, off:off + naux] += dgphi
        return D, Dg

    dchih = [None] * len(cells)
    for c in active:
        r = cells[c]
        ch, gh = chih[c]
        p = pts if np.allclose(r, 0.0) else pts - r[None, :]
        dchih[c] = _dchi(p, ch, gh)
    return dchih


def _lattice_param_drho(chih, dchih, pblocks, pairs, is_gga, npts):
    """dr/dη (+ dgradr/dη for GGA) for one (spin) density, cross-cell pair sum.

    For the physical periodic density r = S_{(a,s)} chi_a P(s-a) chi_s the target
    shell appears in BOTH the bra-image factor chi_a and the ket factor chi_s, so

        dr = S_{(a,s)} [ dchi_a . P(s-a) . chi_s + chi_a . P(s-a) . dchi_s ] .

    ``chih``/``dchih`` are the per-cell AO values and their parameter derivatives
    (:func:`_periodic_grid_param_dchi`); ``pairs`` the ``(a, s, gp)`` bra/ket/block
    positions; ``pblocks`` this spin's per-cell P(h) (block gp is P(s-a))."""
    drho = np.zeros(npts)
    dgrho = np.zeros((3, npts)) if is_gga else None
    for a, s, gp in pairs:
        chi_a, gchi_a = chih[a]
        chi_s, gchi_s = chih[s]
        dchi_a, dgchi_a = dchih[a]
        dchi_s, dgchi_s = dchih[s]
        Ph = pblocks[gp]
        drho += (np.einsum("gm,mn,gn->g", dchi_a, Ph, chi_s, optimize=True)
                 + np.einsum("gm,mn,gn->g", chi_a, Ph, dchi_s, optimize=True))
        if is_gga:
            for cc in range(3):
                dgrho[cc] += (np.einsum("gm,mn,gn->g", dgchi_a[cc], Ph, chi_s, optimize=True)
                              + np.einsum("gm,mn,gn->g", dchi_a, Ph, gchi_s[cc], optimize=True)
                              + np.einsum("gm,mn,gn->g", gchi_a[cc], Ph, dchi_s, optimize=True)
                              + np.einsum("gm,mn,gn->g", chi_a, Ph, dgchi_s[cc], optimize=True))
    return drho, dgrho


@dataclass
class _PeriodicXCFrozen:
    """Frozen on-grid XC data for the periodic explicit-XC gradient term, built
    once at the converged density and reused across free parameters. Holds the
    cross-cell bra/ket pair structure (``active``/``pairs``) so the analytic
    gradient differentiates exactly the density it assembled."""

    mol: Any
    basis: Any
    nbf: int
    pts: np.ndarray
    wts: np.ndarray
    chi: np.ndarray          # (ng, nbf) home AO values
    gchi: Any                # (3, ng, nbf) or None (LDA)
    cells: list              # per-cell Cartesian lattice vectors R_c
    cell_index: list         # per-cell integer index triple (n1, n2, n3)
    chih: list               # per-cell (chi_c, gchi_c) at points r - R_c (active), else None
    pblocks: list            # per-cell real density blocks P(h)
    active: list             # active (ket) cell positions
    pairs: list              # (a, s, gp): bra cell, ket cell, density-block positions
    rho: np.ndarray
    grho: Any
    v_rho: np.ndarray
    v_sigma: Any
    is_gga: bool


def build_periodic_xc_gradient_grid(
    system: Any,
    basis: Any,
    P_real_space: Any,
    functional: str,
    *,
    grid: Any = None,
    lattice_opts: Any = None,
) -> _PeriodicXCFrozen:
    """Assemble the frozen periodic XC grid data (r, gradr, v_r, v_s) at the
    converged real-space density, for :func:`periodic_xc_param_gradient_term`.

    The density is the full cross-cell periodic density
    r = S_{(a,s)} chi_a P(s-a) chi_s that ``build_xc_periodic`` assembles -- bra
    cell a and ket cell s both ranging over the active lattice cells -- so the
    gradient differentiates the same E_xc the SCF minimises. ``functional`` is a
    closed-shell libxc name; LDA and GGA (and global hybrids, whose exact exchange
    lives in the two-electron term, not here) are supported, meta-GGA /
    range-separated / double-hybrid raise. ``P_real_space`` is the converged
    density as a ``LatticeMatrixSet`` (per-cell real blocks P(h)); its cells
    define the lattice sum. ``grid`` defaults to
    ``build_periodic_becke_grid(system)`` and must match the SCF's grid.
    ``lattice_opts`` provides the ket (``cutoff_bohr``) and bra-image
    (``becke_image_radius_bohr``) screens; pass the SCF's options so the active
    cell sets match ``build_xc_periodic`` (defaults to ``LatticeSumOptions()``).
    """
    import vibeqc as vq  # local: vibeqc not importable without a build

    func = vq.Functional(functional, 1)
    if getattr(func, "is_double_hybrid", False) or getattr(func, "is_range_separated", False):
        raise NotImplementedError(
            f"periodic XC gradient: {functional!r} is double-hybrid / "
            f"range-separated; not supported."
        )
    kind = str(getattr(func, "kind", ""))
    if kind not in ("XCKind.LDA", "XCKind.GGA"):
        raise NotImplementedError(
            f"periodic XC gradient supports LDA/GGA(+hybrid) only; {functional!r} "
            f"kind={func.kind} (e.g. meta-GGA) not supported."
        )
    is_gga = kind == "XCKind.GGA"
    if lattice_opts is None:
        lattice_opts = vq.LatticeSumOptions()
    if grid is None:
        grid = vq.build_periodic_becke_grid(system)
    pts = np.asarray(grid.points, dtype=float)
    wts = np.asarray(grid.weights, dtype=float)
    mol = system.unit_cell_molecule()
    nbf = basis.nbasis

    def _ao(p):
        if is_gga:
            v, gx, gy, gz = vq.evaluate_ao_with_gradient(basis, p)
            return np.asarray(v), np.stack([np.asarray(gx), np.asarray(gy), np.asarray(gz)], 0)
        return np.asarray(vq.evaluate_ao(basis, p)), None

    chi, gchi = _ao(pts)
    cells = [np.asarray(P_real_space.cells[c].r_cart, dtype=float).ravel()
             for c in range(len(P_real_space.cells))]
    cell_index = [tuple(int(v) for v in np.asarray(P_real_space.cells[c].index).ravel())
                  for c in range(len(P_real_space.cells))]
    pblocks = [np.asarray(P_real_space.blocks[c], dtype=float)
               for c in range(len(P_real_space.cells))]

    # Cross-cell bra/ket split: mirror build_xc_periodic's active_density_cells.
    has_xc = _density_has_cross_cell(pblocks, cell_index)
    active, pairs = _xc_cross_cell_braket(system, cell_index, cells, lattice_opts, has_xc)

    # Lattice-shifted AO values for the active cells only (chi_c(r) = chi(r - R_c)
    # via shifted grid points, identical to the C++ evaluate_active_shifted_ao).
    chih = [None] * len(cells)
    for c in active:
        r = cells[c]
        chih[c] = (chi, gchi) if np.allclose(r, 0.0) else _ao(pts - r[None, :])

    # r(r) = S_{(a,s)} chi_a P(s-a) chi_s; grad-rho adds the bra + ket AO gradients.
    rho = np.zeros(len(pts))
    grho = np.zeros((3, len(pts))) if is_gga else None
    for a, s, gp in pairs:
        chi_a, gchi_a = chih[a]
        chi_s, gchi_s = chih[s]
        Ph = pblocks[gp]
        chi_aP = chi_a @ Ph
        rho += np.einsum("gm,gm->g", chi_aP, chi_s, optimize=True)
        if is_gga:
            for cc in range(3):
                grho[cc] += (np.einsum("gm,gm->g", chi_aP, gchi_s[cc], optimize=True)
                             + np.einsum("gm,gm->g", gchi_a[cc] @ Ph, chi_s, optimize=True))
    rho = np.where(rho < 0.0, 0.0, rho)  # C++ density floor (periodic_xc.cpp:386)
    if is_gga:
        sigma = np.einsum("cg,cg->g", grho, grho, optimize=True)
        _, v_rho, v_sigma = func.eval_unpolarised(rho, sigma)
        v_rho = np.asarray(v_rho); v_sigma = np.asarray(v_sigma)
    else:
        _, v_rho, _ = func.eval_unpolarised(rho, np.zeros_like(rho))
        v_rho = np.asarray(v_rho); v_sigma = None

    return _PeriodicXCFrozen(mol, basis, nbf, pts, wts, chi, gchi, cells, cell_index,
                             chih, pblocks, active, pairs, rho, grho, v_rho, v_sigma,
                             is_gga)


def periodic_xc_param_gradient_term(
    parametrisation: Any,
    x: np.ndarray,
    spec: Any,
    frozen: _PeriodicXCFrozen,
) -> float:
    """Explicit periodic RKS dE_xc/dη (physical param) for one free spec (P3).

    Uses the frozen grid data from :func:`build_periodic_xc_gradient_grid` and
    the molecular on-grid basis-function derivative dphi_mu/dη
    (:func:`vibeqc.basis_optimization.energy_gradient._param_dphi_on_grid`),
    summed over the libint shells the parameter drives. Returns the derivative
    w.r.t. the *physical* parameter (no optimiser-space chain rule); this is the
    additive XC piece of a periodic KS basis gradient, the one-electron (P1) and
    two-electron (P2) pieces are assembled separately. Exponent and (non-SP)
    coeff params; ``x`` must be the point ``frozen`` was built at.
    """
    import vibeqc as vq  # local: vibeqc not importable without a build
    from .energy_gradient import _shell_atom_maps, _spec_target_shells, _dln_ctilde_dexponent

    atoms = parametrisation.unpack(np.asarray(x, dtype=float))
    f = frozen
    shells, by_atom, atom_syms, ao_off = _shell_atom_maps(vq, f.basis, f.mol)
    targets = _spec_target_shells(spec, parametrisation, shells, by_atom, atom_syms)
    dln_ct = _dln_ctilde_dexponent(vq, atoms, spec) if spec.field == "exponent" else None

    dchih = _periodic_grid_param_dchi(
        vq, f.mol, f.nbf, shells, ao_off, atoms, spec, targets, dln_ct,
        f.is_gga, f.pts, f.cells, f.active, f.chih,
    )
    drho, dgrho = _lattice_param_drho(
        f.chih, dchih, f.pblocks, f.pairs, f.is_gga, len(f.pts),
    )
    contrib = f.v_rho * drho
    if f.is_gga:
        contrib = contrib + 2.0 * f.v_sigma * np.einsum("cg,cg->g", f.grho, dgrho, optimize=True)
    return float(np.sum(f.wts * contrib))


# ---- UKS (open-shell) periodic explicit-XC gradient term -------------------
#
# The spin-polarised counterpart: each spin's physical periodic density is the
# full cross-cell sum r_s(r) = S_{(a,s')} chi_a P_s(s'-a) chi_s' (the same
# bra/ket lattice-cell structure as the RKS path above, with this spin's blocks),
# the polarised libxc evaluation
#
#     v_ra, v_rb, v_saa, v_sab, v_sbb = func.eval_polarised(ra, rb, saa, sab, sbb)
#
# and the gradient term
#
#     dE_xc/dη = S_g w_g [ v_ra dra + v_rb drb
#                          + 2 v_saa gradra.dgradra + 2 v_sbb gradrb.dgradrb
#                          + v_sab (gradra.dgradrb + gradrb.dgradra) ].
#
# dchi/dη is spin-independent, so the per-cell on-grid derivative is computed once
# (``_periodic_grid_param_dchi``) and contracted against each spin's cross-cell
# pair sum (``_lattice_param_drho``). Mirrors the molecular ``_xc_gradient_term_uks``.


@dataclass
class _PeriodicXCFrozenUKS:
    """Frozen on-grid spin-polarised XC data (open-shell counterpart of
    :class:`_PeriodicXCFrozen`); carries the shared cross-cell bra/ket pair
    structure (``active``/``pairs``) the analytic gradient differentiates."""

    mol: Any
    basis: Any
    nbf: int
    pts: np.ndarray
    wts: np.ndarray
    chi: np.ndarray
    gchi: Any
    cells: list
    cell_index: list
    chih: list
    pblocks_a: list          # per-cell P_a(h)
    pblocks_b: list          # per-cell P_b(h)
    active: list             # active (ket) cell positions
    pairs: list              # (a, s, gp): bra cell, ket cell, density-block positions
    rho_a: np.ndarray        # cross-cell r_a (frozen)
    rho_b: np.ndarray        # cross-cell r_b (frozen)
    gr_a: Any                # gradr_a (3, ng) or None
    gr_b: Any
    v_ra: np.ndarray
    v_rb: np.ndarray
    v_saa: Any
    v_sab: Any
    v_sbb: Any
    is_gga: bool


def build_periodic_xc_gradient_grid_uks(
    system: Any,
    basis: Any,
    P_alpha_real: Any,
    P_beta_real: Any,
    functional: str,
    *,
    grid: Any = None,
    lattice_opts: Any = None,
) -> _PeriodicXCFrozenUKS:
    """Open-shell counterpart of :func:`build_periodic_xc_gradient_grid`.

    Assembles the cross-cell per-spin densities r_a, r_b (and gradr_a, gradr_b)
    -- r_s = S_{(a,s')} chi_a P_s(s'-a) chi_s', the same convention as
    ``build_xc_periodic_uks`` -- and the polarised potentials (v_ra, v_rb, v_saa,
    v_sab, v_sbb) at the converged per-spin real-space densities ``P_alpha_real``
    / ``P_beta_real`` (each a ``LatticeMatrixSet``). LDA / GGA (+ hybrid, whose
    exact exchange is in the two-electron term); meta-GGA / range-separated /
    double-hybrid raise. ``lattice_opts`` provides the ket/bra-image screens
    (defaults to ``LatticeSumOptions()``); pass the SCF's options to match
    ``build_xc_periodic_uks``.
    """
    import vibeqc as vq  # local: vibeqc not importable without a build

    func = vq.Functional(functional, 2)  # spin=2 (eval_polarised path)
    if getattr(func, "is_double_hybrid", False) or getattr(func, "is_range_separated", False):
        raise NotImplementedError(
            f"periodic UKS XC gradient: {functional!r} is double-hybrid / "
            f"range-separated; not supported."
        )
    kind = str(getattr(func, "kind", ""))
    if kind not in ("XCKind.LDA", "XCKind.GGA"):
        raise NotImplementedError(
            f"periodic UKS XC gradient supports LDA/GGA(+hybrid) only; "
            f"{functional!r} kind={func.kind} not supported."
        )
    is_gga = kind == "XCKind.GGA"
    if lattice_opts is None:
        lattice_opts = vq.LatticeSumOptions()
    if grid is None:
        grid = vq.build_periodic_becke_grid(system)
    pts = np.asarray(grid.points, dtype=float)
    wts = np.asarray(grid.weights, dtype=float)
    mol = system.unit_cell_molecule()
    nbf = basis.nbasis

    def _ao(p):
        if is_gga:
            v, gx, gy, gz = vq.evaluate_ao_with_gradient(basis, p)
            return np.asarray(v), np.stack([np.asarray(gx), np.asarray(gy), np.asarray(gz)], 0)
        return np.asarray(vq.evaluate_ao(basis, p)), None

    chi, gchi = _ao(pts)
    cells = [np.asarray(P_alpha_real.cells[c].r_cart, dtype=float).ravel()
             for c in range(len(P_alpha_real.cells))]
    cell_index = [tuple(int(v) for v in np.asarray(P_alpha_real.cells[c].index).ravel())
                  for c in range(len(P_alpha_real.cells))]
    pblocks_a = [np.asarray(P_alpha_real.blocks[c], dtype=float)
                 for c in range(len(P_alpha_real.cells))]
    pblocks_b = [np.asarray(P_beta_real.blocks[c], dtype=float)
                 for c in range(len(P_beta_real.cells))]

    # Cross-cell bra/ket split (genuine periodic if either spin carries P(g!=0)).
    has_xc = (_density_has_cross_cell(pblocks_a, cell_index)
              or _density_has_cross_cell(pblocks_b, cell_index))
    active, pairs = _xc_cross_cell_braket(system, cell_index, cells, lattice_opts, has_xc)

    chih = [None] * len(cells)
    for c in active:
        r = cells[c]
        chih[c] = (chi, gchi) if np.allclose(r, 0.0) else _ao(pts - r[None, :])

    rho_a = np.zeros(len(pts)); rho_b = np.zeros(len(pts))
    gr_a = np.zeros((3, len(pts))) if is_gga else None
    gr_b = np.zeros((3, len(pts))) if is_gga else None
    for a, s, gp in pairs:
        chi_a, gchi_a = chih[a]
        chi_s, gchi_s = chih[s]
        Pa = pblocks_a[gp]; Pb = pblocks_b[gp]
        chi_aPa = chi_a @ Pa; chi_aPb = chi_a @ Pb
        rho_a += np.einsum("gm,gm->g", chi_aPa, chi_s, optimize=True)
        rho_b += np.einsum("gm,gm->g", chi_aPb, chi_s, optimize=True)
        if is_gga:
            for cc in range(3):
                gr_a[cc] += (np.einsum("gm,gm->g", chi_aPa, gchi_s[cc], optimize=True)
                             + np.einsum("gm,gm->g", gchi_a[cc] @ Pa, chi_s, optimize=True))
                gr_b[cc] += (np.einsum("gm,gm->g", chi_aPb, gchi_s[cc], optimize=True)
                             + np.einsum("gm,gm->g", gchi_a[cc] @ Pb, chi_s, optimize=True))
    rho_a = np.where(rho_a < 0.0, 0.0, rho_a)  # C++ density floor
    rho_b = np.where(rho_b < 0.0, 0.0, rho_b)
    if is_gga:
        s_aa = np.einsum("cg,cg->g", gr_a, gr_a, optimize=True)
        s_ab = np.einsum("cg,cg->g", gr_a, gr_b, optimize=True)
        s_bb = np.einsum("cg,cg->g", gr_b, gr_b, optimize=True)
        _, v_ra, v_rb, v_saa, v_sab, v_sbb = [
            np.asarray(z) for z in func.eval_polarised(rho_a, rho_b, s_aa, s_ab, s_bb)
        ]
    else:
        z = np.zeros_like(rho_a)
        _, v_ra, v_rb, _, _, _ = [
            np.asarray(q) for q in func.eval_polarised(rho_a, rho_b, z, z, z)
        ]
        v_saa = v_sab = v_sbb = None

    return _PeriodicXCFrozenUKS(mol, basis, nbf, pts, wts, chi, gchi, cells, cell_index,
                               chih, pblocks_a, pblocks_b, active, pairs, rho_a, rho_b,
                               gr_a, gr_b, v_ra, v_rb, v_saa, v_sab, v_sbb, is_gga)


def periodic_xc_param_gradient_term_uks(
    parametrisation: Any,
    x: np.ndarray,
    spec: Any,
    frozen: _PeriodicXCFrozenUKS,
) -> float:
    """Open-shell periodic dE_xc/dη (physical param) for one free spec (P3, UKS).

    The spin-polarised counterpart of :func:`periodic_xc_param_gradient_term`;
    uses the frozen data from :func:`build_periodic_xc_gradient_grid_uks`. dchi/dη is
    spin-independent, so the on-grid derivative is built once and contracted
    against each spin's P_s(h).
    """
    import vibeqc as vq  # local: vibeqc not importable without a build
    from .energy_gradient import _shell_atom_maps, _spec_target_shells, _dln_ctilde_dexponent

    atoms = parametrisation.unpack(np.asarray(x, dtype=float))
    f = frozen
    shells, by_atom, atom_syms, ao_off = _shell_atom_maps(vq, f.basis, f.mol)
    targets = _spec_target_shells(spec, parametrisation, shells, by_atom, atom_syms)
    dln_ct = _dln_ctilde_dexponent(vq, atoms, spec) if spec.field == "exponent" else None

    dchih = _periodic_grid_param_dchi(
        vq, f.mol, f.nbf, shells, ao_off, atoms, spec, targets, dln_ct,
        f.is_gga, f.pts, f.cells, f.active, f.chih,
    )
    npts = len(f.pts)
    drho_a, dgr_a = _lattice_param_drho(
        f.chih, dchih, f.pblocks_a, f.pairs, f.is_gga, npts)
    drho_b, dgr_b = _lattice_param_drho(
        f.chih, dchih, f.pblocks_b, f.pairs, f.is_gga, npts)

    contrib = f.v_ra * drho_a + f.v_rb * drho_b
    if f.is_gga:
        contrib = (contrib
                   + 2.0 * f.v_saa * np.einsum("cg,cg->g", f.gr_a, dgr_a, optimize=True)
                   + 2.0 * f.v_sbb * np.einsum("cg,cg->g", f.gr_b, dgr_b, optimize=True)
                   + f.v_sab * (np.einsum("cg,cg->g", f.gr_a, dgr_b, optimize=True)
                                + np.einsum("cg,cg->g", f.gr_b, dgr_a, optimize=True)))
    return float(np.sum(f.wts * contrib))
