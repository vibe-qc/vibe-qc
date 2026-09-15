"""Periodic explicit-XC basis-parameter gradient (Phase P3) vs frozen-density FD.

Validates :func:`periodic_xc_param_gradient_term` /
:func:`build_periodic_xc_gradient_grid` (the periodic RKS ∂E_xc/∂η):

1. **Convention check.** ``build_periodic_xc_gradient_grid``'s grid density
   reproduces ``build_xc_periodic``'s E_xc on a library-loadable named basis.
   Since the 295f7ada dense-crystal fix BOTH assemble the full cross-cell periodic
   density summing BOTH AOs over lattice cells
   (rho(r) = Sum_a Sum_s chi_a(r) P(s-a) chi_s(r)); the C++ side is independently
   pinned against a NumPy oracle in ``tests/test_periodic_xc_cross_cell.py``, so
   matching the builder to it to ~1e-11 chains the gradient density to the same
   convention the SCF energy minimises.
2. **Gradient.** Analytic ∂E_xc/∂α matches a central finite difference of the
   (Python) E_xc at frozen density, off the reference exponent, for LDA and GGA.
   The FD perturbs the physical exponent; because the reference is the Python
   E_xc (no re-SCF), agreement is to ~1e-9.

Needs a built vibe-qc; skipped otherwise.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

vq = pytest.importorskip("vibeqc")
scipy_linalg = pytest.importorskip("scipy.linalg")

for _needed in ("build_periodic_becke_grid", "build_xc_periodic",
                "evaluate_ao_with_gradient", "compute_overlap_lattice"):
    if not hasattr(vq, _needed):
        pytest.skip(f"{_needed} not in this build", allow_module_level=True)

from vibeqc._vibeqc_core import (  # noqa: E402
    PeriodicSystem, LatticeSumOptions, compute_overlap_lattice,
    compute_kinetic_lattice, bloch_sum,
)
from vibeqc.basis_crystal import parse_crystal_atom_basis_file  # noqa: E402
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.parametrise import (  # noqa: E402
    BasisParametrisation, FreeSpec, Transform,
)
from vibeqc.basis_optimization.periodic_energy_gradient import (  # noqa: E402
    build_periodic_xc_gradient_grid, periodic_xc_param_gradient_term,
    build_periodic_xc_gradient_grid_uks, periodic_xc_param_gradient_term_uks,
)

SRC_H = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP" / "01_H"


def _gamma_density_set(basis, system, opts):
    """Frozen Γ real-space density P(h) = P_Γ in every cell (closed shell)."""
    S_lat = compute_overlap_lattice(basis, system, opts)
    T_lat = compute_kinetic_lattice(basis, system, opts)
    SG = np.real(bloch_sum(S_lat, np.zeros(3)))
    TG = np.real(bloch_sum(T_lat, np.zeros(3)))
    _, C = scipy_linalg.eigh(TG, SG)
    n_occ = system.n_electrons() // 2
    PG = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    P_set = compute_overlap_lattice(basis, system, opts)
    for c in range(len(P_set.cells)):
        P_set.set_block(c, PG)
    return P_set


def _exc_from_grid_uks(frozen, functional):
    """Σ_g w_g exc from a frozen UKS XC grid (recompute exc from the stored
    cross-cell ρ_α/ρ_β/∇ρ_σ)."""
    func = vq.Functional(functional, 2)
    if frozen.is_gga:
        saa = np.einsum("cg,cg->g", frozen.gr_a, frozen.gr_a, optimize=True)
        sab = np.einsum("cg,cg->g", frozen.gr_a, frozen.gr_b, optimize=True)
        sbb = np.einsum("cg,cg->g", frozen.gr_b, frozen.gr_b, optimize=True)
        exc = func.eval_polarised(frozen.rho_a, frozen.rho_b, saa, sab, sbb)[0]
    else:
        z = np.zeros_like(frozen.rho_a)
        exc = func.eval_polarised(frozen.rho_a, frozen.rho_b, z, z, z)[0]
    return float(np.sum(frozen.wts * np.asarray(exc)))


def _exc_from_grid(frozen, functional):
    """Σ_g w_g exc from a frozen XC grid (recompute exc from ρ/σ)."""
    func = vq.Functional(functional, 1)
    if frozen.is_gga:
        sigma = np.einsum("cg,cg->g", frozen.grho, frozen.grho, optimize=True)
        exc, _, _ = func.eval_unpolarised(frozen.rho, sigma)
    else:
        exc, _, _ = func.eval_unpolarised(frozen.rho, np.zeros_like(frozen.rho))
    return float(np.sum(frozen.wts * np.asarray(exc)))


@pytest.mark.parametrize("functional", ["LDA", "PBE"])
def test_density_convention_matches_build_xc_periodic(functional):
    """build_periodic_xc_gradient_grid's cross-cell E_xc == build_xc_periodic E_xc.

    Pins the gradient builder's grid density against ``build_xc_periodic`` on a
    library-loadable named basis (1-D He chain, dense 4-bohr period so the
    bra-image cells contribute). Both now assemble the full cross-cell periodic
    density; with cutoff == becke image radius a single screen governs the ket
    (cutoff_bohr) and bra (min(becke_image_radius_bohr, cutoff_bohr)) cell sets on
    both sides, so they agree to ~1e-11 (the C++ side is independently checked
    against a NumPy oracle in tests/test_periodic_xc_cross_cell.py).
    """
    a = 4.0
    lattice = np.array([[a, 0, 0], [0, 1e6, 0], [0, 0, 1e6]])
    system = PeriodicSystem(1, lattice, [vq.Atom(2, [0.0, 0, 0])], charge=0, multiplicity=1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
    radius = 10.0  # matches build_periodic_becke_grid's default image_radius_bohr
    opts = LatticeSumOptions()
    opts.cutoff_bohr = radius
    opts.becke_image_radius_bohr = radius
    P_set = _gamma_density_set(basis, system, opts)
    grid = vq.build_periodic_becke_grid(system, image_radius_bohr=radius)

    frozen = build_periodic_xc_gradient_grid(
        system, basis, P_set, functional, grid=grid, lattice_opts=opts)
    e_py = _exc_from_grid(frozen, functional)
    e_cpp = float(vq.build_xc_periodic(basis, system, grid, vq.Functional(functional, 1),
                                       P_set, opts).e_xc)
    np.testing.assert_allclose(e_py, e_cpp, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("functional", ["LDA", "PBE"])
def test_xc_gradient_matches_frozen_density_fd(functional):
    """Analytic ∂E_xc/∂α vs central FD of the frozen-density E_xc (off ref)."""
    a = 5.0
    lattice = np.array([[a, 0, 0], [0, 1e6, 0], [0, 0, 1e6]])
    system = PeriodicSystem(
        1, lattice, [vq.Atom(1, [0.0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
        charge=0, multiplicity=1,
    )
    mol = system.unit_cell_molecule()
    opts = LatticeSumOptions()
    grid = vq.build_periodic_becke_grid(system)

    par = BasisParametrisation(
        atoms={"H": parse_crystal_atom_basis_file(SRC_H)},
        free=[FreeSpec("H", 0, 0, "exponent", transform=Transform.LOG)],
    )
    spec = par.free[0]
    x0 = par.pack()
    atoms0 = par.unpack(x0)
    alpha0 = float(atoms0["H"].shells[spec.shell_idx].exponents[spec.prim_idx])

    with TempBasisLibrary() as lib:
        basis0 = vq.BasisSet(mol, lib.write_g94(atoms0, basis_name="p3-ref"))
        P_set = _gamma_density_set(basis0, system, opts)

        frozen0 = build_periodic_xc_gradient_grid(system, basis0, P_set, functional, grid=grid)
        dE_an = periodic_xc_param_gradient_term(par, x0, spec, frozen0)

        def exc_at_alpha(da, tag):
            atoms = copy.deepcopy(atoms0)
            exps = list(atoms["H"].shells[spec.shell_idx].exponents)
            exps[spec.prim_idx] = alpha0 + da
            atoms["H"].shells[spec.shell_idx].exponents = exps
            basis = vq.BasisSet(mol, lib.write_g94(atoms, basis_name=tag))
            fr = build_periodic_xc_gradient_grid(system, basis, P_set, functional, grid=grid)
            return _exc_from_grid(fr, functional)

        h = 1e-4 * alpha0
        dE_fd = (exc_at_alpha(+h, "p3-p") - exc_at_alpha(-h, "p3-m")) / (2.0 * h)

    assert abs(dE_an) > 1e-8  # a real signal, not a trivial near-zero match
    np.testing.assert_allclose(dE_an, dE_fd, atol=1e-9, rtol=1e-5)


# ---------------------------------------------------------------------------
# UKS (open-shell) periodic explicit-XC gradient term
# ---------------------------------------------------------------------------


def _gamma_uks_density_sets(basis, system, opts, na, nb):
    """Frozen Γ per-spin densities P_α(h)=P_α,Γ, P_β(h)=P_β,Γ (occupation 1)."""
    S_lat = compute_overlap_lattice(basis, system, opts)
    T_lat = compute_kinetic_lattice(basis, system, opts)
    SG = np.real(bloch_sum(S_lat, np.zeros(3)))
    TG = np.real(bloch_sum(T_lat, np.zeros(3)))
    _, C = scipy_linalg.eigh(TG, SG)
    PA = C[:, :na] @ C[:, :na].T
    PB = C[:, :nb] @ C[:, :nb].T
    PA_set = compute_overlap_lattice(basis, system, opts)
    PB_set = compute_overlap_lattice(basis, system, opts)
    for c in range(len(PA_set.cells)):
        PA_set.set_block(c, PA)
        PB_set.set_block(c, PB)
    return PA_set, PB_set


@pytest.mark.parametrize("functional", ["LDA", "PBE"])
def test_uks_density_convention_matches_build_xc_periodic_uks(functional):
    """build_periodic_xc_gradient_grid_uks's cross-cell E_xc == build_xc_periodic_uks.

    Open-shell counterpart of the RKS convention check: each spin's physical
    periodic density is the full cross-cell sum (ρ_s = Σ_{a,s'} χ_a P_s(s'−a) χ_s'),
    now assembled by the builder on both sides. Li 1-D chain, cutoff == becke image
    radius (single screen); the C++ side is independently pinned in
    tests/test_periodic_xc_cross_cell.py.
    """
    a = 5.0
    lattice = np.array([[a, 0, 0], [0, 1e6, 0], [0, 0, 1e6]])
    system = PeriodicSystem(1, lattice, [vq.Atom(3, [0.0, 0, 0])], charge=0, multiplicity=2)
    basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
    radius = 10.0
    opts = LatticeSumOptions()
    opts.cutoff_bohr = radius
    opts.becke_image_radius_bohr = radius
    na, nb = 2, 1  # Li: 3 electrons
    PA_set, PB_set = _gamma_uks_density_sets(basis, system, opts, na, nb)
    grid = vq.build_periodic_becke_grid(system, image_radius_bohr=radius)

    frozen = build_periodic_xc_gradient_grid_uks(
        system, basis, PA_set, PB_set, functional, grid=grid, lattice_opts=opts)
    e_py = _exc_from_grid_uks(frozen, functional)
    e_cpp = float(vq.build_xc_periodic_uks(basis, system, grid, vq.Functional(functional, 2),
                                           PA_set, PB_set, opts).e_xc)
    np.testing.assert_allclose(e_py, e_cpp, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("functional", ["LDA", "PBE"])
def test_uks_xc_gradient_matches_frozen_density_fd(functional):
    """Analytic UKS ∂E_xc/∂α vs central FD of the frozen-density E_xc."""
    a = 5.0
    lattice = np.array([[a, 0, 0], [0, 1e6, 0], [0, 0, 1e6]])
    # 2 H per cell, charge -1 -> 3 electrons -> n_alpha=2, n_beta=1 (both nonzero)
    system = PeriodicSystem(
        1, lattice, [vq.Atom(1, [0.0, 0, 0]), vq.Atom(1, [1.4, 0, 0])],
        charge=-1, multiplicity=2,
    )
    mol = system.unit_cell_molecule()
    opts = LatticeSumOptions()
    grid = vq.build_periodic_becke_grid(system)

    par = BasisParametrisation(
        atoms={"H": parse_crystal_atom_basis_file(SRC_H)},
        free=[FreeSpec("H", 0, 0, "exponent", transform=Transform.LOG)],
    )
    spec = par.free[0]
    x0 = par.pack()
    atoms0 = par.unpack(x0)
    alpha0 = float(atoms0["H"].shells[spec.shell_idx].exponents[spec.prim_idx])

    with TempBasisLibrary() as lib:
        basis0 = vq.BasisSet(mol, lib.write_g94(atoms0, basis_name="p3u-ref"))
        PA_set, PB_set = _gamma_uks_density_sets(basis0, system, opts, 2, 1)

        frozen0 = build_periodic_xc_gradient_grid_uks(system, basis0, PA_set, PB_set, functional, grid=grid)
        dE_an = periodic_xc_param_gradient_term_uks(par, x0, spec, frozen0)

        def exc_at_alpha(da, tag):
            atoms = copy.deepcopy(atoms0)
            exps = list(atoms["H"].shells[spec.shell_idx].exponents)
            exps[spec.prim_idx] = alpha0 + da
            atoms["H"].shells[spec.shell_idx].exponents = exps
            basis = vq.BasisSet(mol, lib.write_g94(atoms, basis_name=tag))
            fr = build_periodic_xc_gradient_grid_uks(system, basis, PA_set, PB_set, functional, grid=grid)
            # E_xc from the builder's stored cross-cell ρ_α/ρ_β (the same density
            # the analytic gradient differentiates) -- not a home-bra recompute.
            return _exc_from_grid_uks(fr, functional)

        h = 1e-4 * alpha0
        dE_fd = (exc_at_alpha(+h, "p3u-p") - exc_at_alpha(-h, "p3u-m")) / (2.0 * h)

    assert abs(dE_an) > 1e-8
    np.testing.assert_allclose(dE_an, dE_fd, atol=1e-9, rtol=1e-5)
