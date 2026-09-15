"""Bloch-summed ∂S(k)/∂α and ∂T(k)/∂α vs a Bloch-summed integral FD (Phase P1).

Validates :func:`bloch_summed_one_electron_exponent_derivatives` — the periodic
one-electron exponent derivatives that Bloch-sum the molecular
overlap/kinetic exponent-derivative bindings between a home-cell shell and its
image at each lattice vector R (see
``docs/basisset_dev/PERIODIC_GRADIENT_DESIGN.md`` Phase P1).

The reference is a central finite difference of the *Bloch-summed* lattice
integral at a fixed k,

    ∂M(k)/∂α ≈ [bloch_sum(M_lat(α+h), k) − bloch_sum(M_lat(α−h), k)] / 2h ,

with the perturbed basis built directly from :class:`ShellInfo` (so the tiny
exponent step is exact — a ``.g94`` round-trip would quantise it, per the
molecular hard-won lesson). Needs a built vibe-qc (the integral bindings);
skipped otherwise.

Coverage: a symmetric 1D chain (real S(k); k = 0 and k ≠ 0; both primitives of
a contracted shell, so the contracted-renormalisation response is exercised), an
asymmetric two-atom 1D cell with s + p shells (complex S(k); l = 1; a target
list spanning two atoms), and a 3D simple-cubic cell (the Bloch loop over a
genuinely 3D image set).
"""

from __future__ import annotations

import numpy as np
import pytest

vq = pytest.importorskip("vibeqc")

for _needed in (
    "overlap_exponent_derivative",
    "kinetic_exponent_derivative",
    "compute_overlap_lattice",
    "compute_kinetic_lattice",
    "bloch_sum",
):
    if not hasattr(vq, _needed):
        pytest.skip(f"{_needed} not in this build", allow_module_level=True)

from vibeqc.basis_optimization.periodic_energy_gradient import (  # noqa: E402
    bloch_summed_one_electron_exponent_derivatives,
)


def _decoupled_1d(a):
    """1D lattice with huge perpendicular vectors (images only along x)."""
    return np.array([[a, 0.0, 0.0], [0.0, 1.0e6, 0.0], [0.0, 0.0, 1.0e6]])


def _fd_bloch_derivative(make_basis, system, opts, k, latfn, alpha, h):
    """Central FD of bloch_sum(latfn(basis(α±h)), k) w.r.t. the exponent α."""
    mp = vq.bloch_sum(latfn(make_basis(alpha + h), system, opts), k)
    mm = vq.bloch_sum(latfn(make_basis(alpha - h), system, opts), k)
    return (np.asarray(mp) - np.asarray(mm)) / (2.0 * h)


def _assert_matches(make_basis, system, target_shells, prim_idx, alpha,
                    kpoints, opts=None, *, atol=1e-7):
    """Analytic dS(k)/dα, dT(k)/dα vs Bloch-summed integral FD at each k."""
    if opts is None:
        opts = vq.LatticeSumOptions()
    basis = make_basis(alpha)
    dS_an, dT_an = bloch_summed_one_electron_exponent_derivatives(
        system, basis, target_shells, prim_idx, kpoints, lattice_opts=opts
    )
    h = 1e-5
    for ik, k in enumerate(np.atleast_2d(kpoints)):
        dS_fd = _fd_bloch_derivative(
            make_basis, system, opts, k, vq.compute_overlap_lattice, alpha, h
        )
        dT_fd = _fd_bloch_derivative(
            make_basis, system, opts, k, vq.compute_kinetic_lattice, alpha, h
        )
        np.testing.assert_allclose(dS_an[ik], dS_fd, atol=atol, rtol=1e-5)
        np.testing.assert_allclose(dT_an[ik], dT_fd, atol=atol, rtol=1e-5)


def test_symmetric_chain_contracted_s_both_primitives():
    """Symmetric 1D H chain, one contracted (2-primitive) s shell.

    Real S(k); differentiating either primitive exercises the
    contracted-renormalisation coupling that makes even the home (R=0) block
    derivative non-trivial. ``target_shells=[0]`` (one shell, one atom)."""
    a = 2.3
    system = vq.PeriodicSystem(
        1, _decoupled_1d(a), [vq.Atom(1, [0.0, 0.0, 0.0])], charge=0, multiplicity=2
    )
    mol = system.unit_cell_molecule()
    coeffs = [0.6, 0.4]
    fixed_exp = 0.45  # the *other* primitive, held fixed

    def make_basis_p0(alpha):
        sh = vq.ShellInfo(0, 0, True, [alpha, fixed_exp], coeffs, [0.0, 0.0, 0.0])
        return vq.BasisSet(mol, [sh], "p1-sym", False)

    kpoints = np.array([[0.0, 0.0, 0.0], [0.31, 0.0, 0.0], [0.72, 0.0, 0.0]])
    # off any "reference" value: differentiate at α = 1.37
    _assert_matches(make_basis_p0, system, [0], 0, 1.37, kpoints)

    def make_basis_p1(beta):
        sh = vq.ShellInfo(0, 0, True, [1.37, beta], coeffs, [0.0, 0.0, 0.0])
        return vq.BasisSet(mol, [sh], "p1-sym", False)

    _assert_matches(make_basis_p1, system, [0], 1, 0.45, kpoints)


def test_asymmetric_two_atom_cell_with_p_shell():
    """Asymmetric 1D two-atom cell: s+p on atom 0, s on atom 1.

    S(k) is genuinely complex (S(R) != S(-R)); the p shell exercises l=1
    multi-AO blocks. Differentiate the s exponent on atom 0 — a single target
    shell — at several k including ones with a large imaginary part."""
    a = 3.0
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [1.1, 0.0, 0.0])]
    system = vq.PeriodicSystem(1, _decoupled_1d(a), atoms, charge=0, multiplicity=1)
    mol = system.unit_cell_molecule()

    def make_basis(alpha):
        shells = [
            vq.ShellInfo(0, 0, True, [alpha, 0.5], [0.6, 0.4], [0.0, 0.0, 0.0]),
            vq.ShellInfo(0, 1, True, [0.8], [1.0], [0.0, 0.0, 0.0]),
            vq.ShellInfo(1, 0, True, [0.9], [1.0], [1.1, 0.0, 0.0]),
        ]
        return vq.BasisSet(mol, shells, "p1-asym", False)

    kpoints = np.array([[0.4, 0.0, 0.0], [0.93, 0.0, 0.0]])
    # confirm the cell really is asymmetric -> complex S(k)
    basis = make_basis(1.5)
    s_lat = vq.compute_overlap_lattice(basis, system, vq.LatticeSumOptions())
    assert np.max(np.abs(np.asarray(vq.bloch_sum(s_lat, kpoints[0])).imag)) > 1e-2
    _assert_matches(make_basis, system, [0], 0, 1.5, kpoints)


def test_two_atom_shared_element_target_list():
    """Two symmetry-equivalent H atoms per cell sharing one free exponent.

    The free exponent drives the s shell on *both* atoms, so the molecular
    target list is two home shells ([0, 1]); the lattice sum couples the home
    and image copies of both. Validates the multi-target summation."""
    a = 4.0
    atoms = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [2.0, 0.0, 0.0])]
    system = vq.PeriodicSystem(1, _decoupled_1d(a), atoms, charge=0, multiplicity=1)
    mol = system.unit_cell_molecule()

    def make_basis(alpha):
        shells = [
            vq.ShellInfo(0, 0, True, [alpha], [1.0], [0.0, 0.0, 0.0]),
            vq.ShellInfo(1, 0, True, [alpha], [1.0], [2.0, 0.0, 0.0]),
        ]
        return vq.BasisSet(mol, shells, "p1-shared", False)

    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    _assert_matches(make_basis, system, [0, 1], 0, 1.1, kpoints)


def test_simple_cubic_3d():
    """3D simple-cubic He, single s function — the Bloch loop over a 3D image
    set. A tight lattice cutoff keeps the image count (and so the test) cheap."""
    a = 4.2
    lattice = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]])
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])], charge=0, multiplicity=1
    )
    mol = system.unit_cell_molecule()
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 8.0

    def make_basis(alpha):
        sh = vq.ShellInfo(0, 0, True, [alpha, 0.6], [0.5, 0.5], [0.0, 0.0, 0.0])
        return vq.BasisSet(mol, [sh], "p1-sc", False)

    kpoints = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.3, 0.3, 0.3]])
    _assert_matches(make_basis, system, [0], 0, 1.25, kpoints, opts=opts)
