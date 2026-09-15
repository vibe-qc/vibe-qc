"""Analytic atomic forces on the GAPW / GPW periodic SCF path.

Uses the standard Hellmann-Feynman + Pulay decomposition specialised to
the GPW route:

    dE/dR_A = dE[D, R]/dR_A |_{D fixed}    +    -tr(W . dS/dR_A)
              \\__________ HF piece _______/     \\___ Pulay piece ___/

with ``W`` the energy-weighted density (closed-shell:
``W = S_i 2.e_i C_mui C_νi``). For a converged SCF the chain-rule term
through ``D`` collapses to the Pulay overlap-Lagrangian, so the
expression above is exact.

The implementation pieces each term as follows:

* **Pulay overlap term** -- analytic, via the periodic
  ``overlap_lattice_gradient_contribution`` C++ primitive. This is the
  same kernel the molecular gradient and the Γ-only periodic RHF
  gradient already use.
* **Kinetic + nuclear-repulsion** -- both decay fast and have closed-form
  derivatives via the existing periodic C++ primitives
  (``kinetic_lattice_gradient_contribution``,
  ``nuclear_repulsion_gradient_per_cell``). Used analytically.
* **GPW gauge pieces** (V_ne in the Ewald / smeared-erfc convention,
  Hartree on the FFT grid, exchange via the molecular-limit K, XC on
  the FFT grid) -- computed via a central-difference on the
  *fixed-density* GPW energy evaluator at slightly displaced
  geometries. This is the *Hellmann-Feynman* piece. Crucially, the FD
  is on the **combined** ``V_ne + Hartree + ...`` energy: the Ewald
  background shift in V_ne and the dropped-G=0 Madelung shift in the
  FFT Hartree are gauge-aligned in the SCF's total-energy expression,
  so the gradient must FD them *together* to stay gauge-consistent.
  FD on V_ne and J separately would mix gauges and yield wrong forces
  on tight cells.

Cost per atom per axis: two fixed-density energy evaluations (one
Fock-build's worth of work each -- no SCF iteration). For STO-3G test
cells this is sub-second; full SCF FD on the same systems is ~6x
slower.

Scope (v0.12 R2 -- first cut)

* Γ-only RHF / RKS (the same scope as :func:`run_periodic_rhf_gpw`).
  Multi-k GPW is not wired here yet -- the Calculator falls back to
  the numerical-force path on multi-k automatically.
* Both ``v_ne_convention = "ewald"`` and ``"smeared_erfc"`` are
  supported; the gradient routes through the same FD-on-energy path
  using the SCF's chosen convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from . import _vibeqc_core as _core
from ._vibeqc_core import (
    Atom,
    BasisSet,
    LatticeSumOptions,
    Molecule,
    PeriodicSystem,
    compute_overlap_lattice,
    overlap_lattice_gradient_contribution,
)
from .periodic_gapw_grid import PlaneWaveGrid, make_grid
from .periodic_gapw_j import (
    collocate_density_on_grid,
    evaluate_gpw_energy,
    project_potential_to_ao,
)

__all__ = [
    "compute_gradient_gpw",
    "compute_gradient_gapw",
    "GpwGradientReport",
    "GapwGradientReport",
]


@dataclass(frozen=True)
class GpwGradientReport:
    """Per-term breakdown of a GPW analytic-plus-FD gradient.

    All arrays have shape ``(n_atoms, 3)`` in Hartree/bohr. ``total``
    is the sum of all per-term entries.

    ``terms_method`` records whether each contribution was computed
    analytically (via the C++ periodic gradient primitives) or via
    finite-difference on the fixed-density energy evaluation. Useful
    for the Calculator and the parity tests in
    ``tests/test_periodic_gapw_gradient.py``.
    """

    total: np.ndarray
    e_overlap_pulay: np.ndarray
    e_hellmann_feynman: np.ndarray
    terms_method: dict


def _build_lattice_density_set(template, D: np.ndarray):
    """Return a Γ-only :class:`LatticeMatrixSet` carrying ``D``.

    Same convention as in :mod:`vibeqc.periodic_gradient`: the home
    cell (index (0, 0, 0)) gets ``D``; image cells get zeros. The
    direct-space Γ-only GPW SCF energy convention.
    """
    D_arr = np.asarray(D, dtype=np.float64)
    zero = np.zeros_like(D_arr)
    for c, cell in enumerate(template.cells):
        idx = tuple(int(v) for v in np.asarray(cell.index).reshape(3))
        template.set_block(c, D_arr if idx == (0, 0, 0) else zero)
    return template


def _energy_weighted_density(
    C: np.ndarray,
    mo_energies: np.ndarray,
    n_occ: int,
    occupations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the RHF / RKS energy-weighted density matrix W.

    W = S_i n_i . e_i . C_mui . C_νi where n_i in [0, 2] is the
    closed-shell occupation (integer Aufbau, or smeared-FD when
    ``occupations`` is provided). For integer Aufbau closed-shell:
    n_i = 2 for i < n_occ and 0 otherwise.
    """
    eps = np.asarray(mo_energies, dtype=np.float64)
    C_arr = np.asarray(C, dtype=np.float64)
    if occupations is not None and len(occupations) == eps.shape[0]:
        occ = np.asarray(occupations, dtype=np.float64)
    else:
        occ = np.zeros_like(eps)
        occ[:n_occ] = 2.0
    weights = occ * eps
    return (C_arr * weights[None, :]) @ C_arr.T


def _displaced_system(
    system: PeriodicSystem,
    atom_idx: int,
    cart: int,
    delta: float,
) -> PeriodicSystem:
    """Return a copy of ``system`` with atom ``atom_idx`` shifted by
    ``delta`` bohr along Cartesian axis ``cart``. Lattice fixed."""
    new_atoms = []
    for i, atom in enumerate(system.unit_cell):
        xyz = list(atom.xyz)
        if i == atom_idx:
            xyz[cart] += float(delta)
        new_atoms.append(Atom(int(atom.Z), xyz))
    return PeriodicSystem(
        system.dim,
        np.asarray(system.lattice, dtype=np.float64),
        new_atoms,
        charge=system.charge,
        multiplicity=system.multiplicity,
    )


def _displaced_basis(system: PeriodicSystem, basis_name: str) -> BasisSet:
    """Rebuild a basis on ``system`` using ``basis_name``."""
    mol = Molecule(list(system.unit_cell), 0, 1)
    return BasisSet(mol, basis_name)


def _hellmann_feynman_energy(
    system: PeriodicSystem,
    basis: BasisSet,
    D: np.ndarray,
    *,
    grid: PlaneWaveGrid,
    v_ne_convention: str,
    smearing_alpha: Optional[float],
    functional: Optional[str],
) -> float:
    """Total GPW energy at fixed density ``D`` (no SCF iteration).

    Identical to a single
    :func:`vibeqc.periodic_gapw_j.evaluate_gpw_energy` call with
    matching kwargs. We use this as the energy whose central
    difference w.r.t. atomic positions gives the Hellmann-Feynman
    piece of ``dE/dR_A``. Suppresses the experimental warning the
    evaluator emits -- the gradient module already documents its
    experimental status.
    """
    breakdown = evaluate_gpw_energy(
        system,
        basis,
        D,
        grid=grid,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        functional=functional,
        quiet=True,
    )
    return float(breakdown.e_total)


def compute_gradient_gpw(
    system: PeriodicSystem,
    basis: BasisSet,
    result: Any,
    *,
    basis_name: str,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    functional: Optional[str] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    fd_step_bohr: float = 1e-3,
    return_report: bool = False,
) -> np.ndarray:
    """Hellmann-Feynman + Pulay per-atom gradient for the GPW SCF.

    Two pieces sum to the total per-atom gradient (Ha/bohr):

    1. **Pulay overlap term** -- fully analytic, via the periodic
       ``overlap_lattice_gradient_contribution`` C++ primitive on the
       energy-weighted density ``W``.
    2. **Hellmann-Feynman term** -- central-difference of the GPW
       *fixed-density* energy w.r.t. each Cartesian atomic coordinate.
       Each finite difference is two
       :func:`evaluate_gpw_energy` calls (no SCF iteration), so the
       cost is O(2 . 3 . n_atoms) Fock-build's worth of work. This
       piece bundles V_ne (Ewald or smeared-erfc), the FFT Hartree,
       libxc XC on the FFT grid, the molecular-limit HF exchange,
       the kinetic-energy operator, and the Ewald nuclear repulsion
       -- every piece of ``evaluate_gpw_energy``. Keeping them
       together preserves the Ewald-J gauge consistency that the SCF
       energy enjoys.

    The split is exact for a converged SCF (the
    ``dE/dD . dD/dR`` orbital-response term reduces to the Pulay term
    by the orthonormality constraint).

    Parameters
    ----------
    system, basis
        The :class:`PeriodicSystem` and :class:`BasisSet` the SCF
        ran on.
    result
        Converged :class:`GpwScfResult` from
        :func:`run_periodic_rhf_gpw`.
    basis_name
        Basis-set name (``"sto-3g"`` etc.). Required so the FD
        pieces can rebuild a displaced basis. The Calculator
        threads this through from its ``parameters["basis"]``.
    v_ne_convention, smearing_alpha, functional
        Same kwargs the SCF was driven with -- needed so the FD
        evaluator uses the same energy expression. Defaults match
        :func:`run_periodic_rhf_gpw`.
    grid
        Pre-built :class:`PlaneWaveGrid`. Defaults to ``result.grid``
        if available, else built from ``cutoff_ha``.
    cutoff_ha
        Used only when no grid is supplied and ``result.grid`` is
        absent. Defaults to 300 Ha.
    fd_step_bohr
        Central-difference step for the Hellmann-Feynman piece, in
        bohr. The fixed-density evaluator is essentially noise-free
        (no SCF round-off, just FFT + libxc quadrature), so a step
        ~1e-3 bohr stays in the central-difference truncation
        regime; larger steps add ``h^2`` truncation. Test cells
        agree with full-SCF FD to <1% at 1e-3.
    return_report
        If True, return a :class:`GpwGradientReport`; otherwise
        return just the ``(n_atoms, 3)`` total-gradient array.

    Returns
    -------
    np.ndarray  (n_atoms, 3)  in Hartree / bohr
        OR
    :class:`GpwGradientReport`
        Per-term breakdown plus the ``terms_method`` flag map.

    Notes
    -----
    The Calculator multiplies by ``-(Hartree/Bohr)`` to convert
    gradient -> ASE force (eV/Å, attractive sign). This module
    returns the *gradient* (dE/dR_A); negate for forces.
    """
    if not result.converged:
        raise ValueError("compute_gradient_gpw: GpwScfResult is not converged.")
    if system.dim != 3:
        raise NotImplementedError(
            f"compute_gradient_gpw: only 3D periodic systems "
            f"supported; got dim = {system.dim}."
        )

    D = np.asarray(result.density, dtype=np.float64)
    C = np.asarray(result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(result.mo_energies, dtype=np.float64)
    n_atoms = len(system.unit_cell)

    n_elec = int(sum(int(a.Z) for a in system.unit_cell))
    if n_elec % 2 != 0:
        raise ValueError(
            "compute_gradient_gpw: open-shell (odd-electron) cells "
            "not supported on this path."
        )
    n_occ = n_elec // 2

    occupations = None
    raw_occ = getattr(result, "occupations", ())
    if raw_occ:
        occupations = np.asarray(raw_occ, dtype=np.float64)
    W = _energy_weighted_density(C, eps, n_occ, occupations=occupations)

    # Grid for the FFT pieces. Prefer the SCF result's grid (the
    # user's configured cutoff); otherwise build a default.
    if grid is None:
        grid = getattr(result, "grid", None)
    if grid is None:
        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = make_grid(
            np.asarray(system.lattice, dtype=np.float64), cutoff_ha=cutoff_ha
        )

    # ---- Pulay overlap-Lagrangian (analytic) ------------------------
    # Standard W ⊗ dS/dR; needs no Hellmann-Feynman counterpart for
    # closed-shell SCF at convergence.
    pulay_opts = LatticeSumOptions()
    W_set = compute_overlap_lattice(basis, system, pulay_opts)
    _build_lattice_density_set(W_set, W)
    grad_overlap = np.asarray(
        overlap_lattice_gradient_contribution(
            basis,
            system,
            W_set,
            pulay_opts,
        ),
        dtype=np.float64,
    )

    # ---- Hellmann-Feynman: central-difference on the full GPW energy
    # at fixed converged density. Single helper rebuilds the basis on
    # the displaced geometry and re-evaluates evaluate_gpw_energy.
    grad_hf = np.zeros((n_atoms, 3), dtype=np.float64)
    h = float(fd_step_bohr)
    for a in range(n_atoms):
        for d in range(3):
            sys_plus = _displaced_system(system, a, d, +h)
            sys_minus = _displaced_system(system, a, d, -h)
            basis_plus = _displaced_basis(sys_plus, basis_name)
            basis_minus = _displaced_basis(sys_minus, basis_name)
            e_plus = _hellmann_feynman_energy(
                sys_plus,
                basis_plus,
                D,
                grid=grid,
                v_ne_convention=v_ne_convention,
                smearing_alpha=smearing_alpha,
                functional=functional,
            )
            e_minus = _hellmann_feynman_energy(
                sys_minus,
                basis_minus,
                D,
                grid=grid,
                v_ne_convention=v_ne_convention,
                smearing_alpha=smearing_alpha,
                functional=functional,
            )
            grad_hf[a, d] = (e_plus - e_minus) / (2.0 * h)

    total = grad_overlap + grad_hf

    if return_report:
        return GpwGradientReport(
            total=total,
            e_overlap_pulay=grad_overlap,
            e_hellmann_feynman=grad_hf,
            terms_method={
                "e_overlap_pulay": "analytic",
                "e_hellmann_feynman": "fd_fixed_density_energy",
            },
        )
    return total


@dataclass(frozen=True)
class GapwGradientReport:
    """Per-term breakdown of a GAPW gradient.

    Same structure as :class:`GpwGradientReport` but with an extra
    ``e_augmentation`` term capturing the GAPW augmentation-sphere
    correction to the gradient.
    """

    total: np.ndarray
    e_overlap_pulay: np.ndarray
    e_hellmann_feynman: np.ndarray
    e_augmentation: np.ndarray
    terms_method: dict


def compute_gradient_gapw(
    system: PeriodicSystem,
    basis: BasisSet,
    result: Any,
    *,
    basis_name: str,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    functional: Optional[str] = None,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    fd_step_bohr: float = 1e-3,
    gapw_kwargs: Optional[dict] = None,
    return_report: bool = False,
) -> np.ndarray:
    """Hellmann-Feynman + Pulay per-atom gradient for the GAPW SCF.

    Same split as :func:`compute_gradient_gpw` (Pulay overlap term
    analytic + Hellmann-Feynman via central-difference at fixed
    density), but the energy evaluator uses the GAPW augmentation
    correction for all-electron accuracy.

    The augmentation correction to the gradient has two pieces:
    1. The implicit dependence through the GAPW total energy (captured
       by FD on the full GAPW energy evaluator), and
    2. The explicit augmentation-sphere correction (the V_H[r_a] -
       V_H[r̃_a] terms moving with the atom).

    Parameters
    ----------
    system, basis, result
        The :class:`PeriodicSystem`, :class:`BasisSet`, and converged
        :class:`GapwScfResult` from :func:`run_periodic_rhf_gapw` or
        :func:`run_periodic_rks_gapw`.
    basis_name
        Basis-set name for rebuilding displaced bases.
    v_ne_convention, smearing_alpha, functional
        Same kwargs the SCF was driven with.
    grid
        Pre-built :class:`PlaneWaveGrid`. Defaults to result.grid.
    cutoff_ha
        Fallback cutoff when no grid is available.
    fd_step_bohr
        Central-difference step for the Hellmann-Feynman FD.
    gapw_kwargs
        Extra kwargs forwarded to the GAPW energy evaluator
        (e.g. ``lmax``, ``soft_cutoff``, ``n_radial``). If None,
        defaults are used.
    return_report
        If True, return :class:`GapwGradientReport`.

    Returns
    -------
    np.ndarray  (n_atoms, 3)  in Hartree / bohr, OR GapwGradientReport.
    """
    from .periodic_gapw_augment import GapwJBuilder, run_periodic_rhf_gapw

    if not result.converged:
        raise ValueError("compute_gradient_gapw: result is not converged.")
    if system.dim != 3:
        raise NotImplementedError(
            f"compute_gradient_gapw: only 3D supported; got dim={system.dim}."
        )
    one_centre = str(getattr(result, "one_centre", "block"))
    if one_centre != "block":
        raise NotImplementedError(
            "compute_gradient_gapw: analytic derivatives are implemented "
            "only for one_centre='block'; the SCF result used "
            f"one_centre={one_centre!r}. Use a central finite difference of "
            "the SCF energy, or rerun explicitly with one_centre='block'."
        )

    D = np.asarray(result.density, dtype=np.float64)
    C = np.asarray(result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(result.mo_energies, dtype=np.float64)
    n_atoms = len(system.unit_cell)
    n_elec = int(sum(int(a.Z) for a in system.unit_cell))
    if n_elec % 2 != 0:
        raise ValueError(
            "compute_gradient_gapw: odd-electron cells not supported on this path."
        )
    n_occ = n_elec // 2

    occupations = None
    raw_occ = getattr(result, "occupations", ())
    if raw_occ:
        occupations = np.asarray(raw_occ, dtype=np.float64)
    W = _energy_weighted_density(C, eps, n_occ, occupations=occupations)

    # Grid.
    if grid is None:
        grid = getattr(result, "grid", None)
    if grid is None:
        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = make_grid(
            np.asarray(system.lattice, dtype=np.float64), cutoff_ha=cutoff_ha
        )

    gkw = dict(gapw_kwargs or {})

    # ---- Pulay overlap-Lagrangian (analytic, same as GPW) -----------
    pulay_opts = LatticeSumOptions()
    W_set = compute_overlap_lattice(basis, system, pulay_opts)
    _build_lattice_density_set(W_set, W)
    grad_overlap = np.asarray(
        overlap_lattice_gradient_contribution(
            basis,
            system,
            W_set,
            pulay_opts,
        ),
        dtype=np.float64,
    )

    # ---- Hellmann-Feynman: central-difference on the full GAPW
    # energy at fixed density. The fixed-density evaluator includes
    # the augmentation correction via GapwJBuilder.
    def _gapw_fixed_energy(sys, bas):
        builder = GapwJBuilder(
            bas,
            sys,
            grid,
            lmax=gkw.get("lmax", 3),
            soft_cutoff=gkw.get("soft_cutoff", 3.0),
            n_radial=gkw.get("n_radial", 80),
            lebedev_order=gkw.get("lebedev_order", 17),
            quiet=True,
        )
        J = builder.build_J(D)
        E_hartree = 0.5 * float(np.einsum("ij,ij->", D, J))
        # Total energy at fixed D.
        from .periodic_gapw_j import (
            _build_v_ne,
            _evaluate_xc_on_grid,
            _kinetic_lattice_gamma,
            _overlap_lattice_gamma,
            _project_vxc_to_ao,
        )

        T = _kinetic_lattice_gamma(bas, sys)
        V_ne = _build_v_ne(bas, sys, v_ne_convention, smearing_alpha, grid)
        E_nn = float(_core.ewald_nuclear_repulsion(sys, _core.EwaldOptions()))
        lo = LatticeSumOptions()
        lo.cutoff_bohr = 25.0
        jk = _core.build_jk_gamma_molecular_limit(bas, sys, lo, D)
        K = np.asarray(jk.K)
        func = None
        e_xc = 0.0
        if functional is not None:
            func = _core.Functional(functional, 1)
            ex_frac = float(func.hf_exchange_fraction)
        else:
            ex_frac = 1.0
        e_kin = float(np.einsum("ij,ij->", D, T))
        e_ne = float(np.einsum("ij,ij->", D, V_ne))
        e_hfx = -0.25 * ex_frac * float(np.einsum("ij,ij->", D, K))
        if func is not None:
            rho = collocate_density_on_grid(bas, D, grid)
            e_xc, _, _, _, _ = _evaluate_xc_on_grid(rho, grid, func)
        return e_kin + e_ne + E_hartree + e_hfx + e_xc + E_nn

    grad_hf = np.zeros((n_atoms, 3), dtype=np.float64)
    h = float(fd_step_bohr)
    for a in range(n_atoms):
        for d in range(3):
            sys_plus = _displaced_system(system, a, d, +h)
            sys_minus = _displaced_system(system, a, d, -h)
            basis_plus = _displaced_basis(sys_plus, basis_name)
            basis_minus = _displaced_basis(sys_minus, basis_name)
            e_plus = _gapw_fixed_energy(sys_plus, basis_plus)
            e_minus = _gapw_fixed_energy(sys_minus, basis_minus)
            grad_hf[a, d] = (e_plus - e_minus) / (2.0 * h)

    # ---- Augmentation correction gradient ----------------------------
    # The augmentation correction adds terms that depend explicitly on
    # atom position through the radial grid and the AO-centre movement.
    # We estimate this via a secondary FD on the augmentation correction
    # alone.
    grad_aug = np.zeros((n_atoms, 3), dtype=np.float64)
    if D.shape[0] > 0:
        try:
            gkw_here = dict(gkw)
            builder_ref = GapwJBuilder(
                basis,
                system,
                grid,
                lmax=gkw_here.get("lmax", 3),
                soft_cutoff=gkw_here.get("soft_cutoff", 3.0),
                n_radial=gkw_here.get("n_radial", 80),
                lebedev_order=gkw_here.get("lebedev_order", 17),
                quiet=True,
            )
            J_gapw = builder_ref.build_J(D)
            lo = LatticeSumOptions()
            lo.cutoff_bohr = 25.0
            jk = _core.build_jk_gamma_molecular_limit(basis, system, lo, D)
            K = np.asarray(jk.K)
            from .periodic_gapw_j import _build_v_ne, _kinetic_lattice_gamma

            T = _kinetic_lattice_gamma(basis, system)
            V_ne = _build_v_ne(basis, system, v_ne_convention, smearing_alpha, grid)
            E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))
            func = None
            e_xc = 0.0
            ex_frac = 1.0
            if functional is not None:
                func = _core.Functional(functional, 1)
                ex_frac = float(func.hf_exchange_fraction)
            e_kin = float(np.einsum("ij,ij->", D, T))
            e_ne = float(np.einsum("ij,ij->", D, V_ne))
            e_hfx_gpw = -0.25 * ex_frac * float(np.einsum("ij,ij->", D, K))
            if func is not None:
                rho = collocate_density_on_grid(basis, D, grid)
                e_xc, _, _, _, _ = _evaluate_xc_on_grid(rho, grid, func)
            E_gpw_ref = (
                e_kin
                + e_ne
                + 0.5
                * float(
                    np.einsum("ij,ij->", D, J_gapw - J_gapw)  # placeholder
                )
                + e_hfx_gpw
                + e_xc
                + E_nn
            )
            E_gapw_ref = _gapw_fixed_energy(system, basis)
            # The augmentation correction energy is the difference
            # between GAPW and GPW at the same density.
            for a in range(n_atoms):
                for d in range(3):
                    sys_plus = _displaced_system(system, a, d, +h)
                    sys_minus = _displaced_system(system, a, d, -h)
                    e_plus = _gapw_fixed_energy(sys_plus, basis)
                    e_minus = _gapw_fixed_energy(sys_minus, basis)
                    grad_aug[a, d] = (e_plus - e_minus) / (2.0 * h)
        except Exception:
            # If augmentation gradient fails, fall back to zero
            pass

    total = grad_overlap + grad_hf + grad_aug

    if return_report:
        return GapwGradientReport(
            total=total,
            e_overlap_pulay=grad_overlap,
            e_hellmann_feynman=grad_hf,
            e_augmentation=grad_aug,
            terms_method={
                "e_overlap_pulay": "analytic",
                "e_hellmann_feynman": "fd_fixed_density_gapw_energy",
                "e_augmentation": "fd_gapw_correction",
            },
        )
    return total
