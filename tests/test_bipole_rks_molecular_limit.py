"""Gamma BIPOLE-RKS molecular-limit regressions for GitLab #102."""

from __future__ import annotations

import math

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    InitialGuess,
    LatticeSumOptions,
    PeriodicKSOptions,
    compute_overlap_lattice,
    ewald_nuclear_repulsion,
    make_lattice_matrix_set,
    monkhorst_pack,
)
from vibeqc.bipole_ext_el_pole import (
    crystal_default_ewald_alpha,
    crystal_ewald_reciprocal_cutoff,
)
from vibeqc.pbc_bipole_common import (
    _crystal_ewald_options,
    _density_set_gamma_or_lattice,
)
from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks


def test_gamma_xc_density_adapter_preserves_overlap_template():
    """The XC adapter must not replace the live overlap blocks with D."""
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 50.0,
        [vq.Atom(2, [25.0, 25.0, 25.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lattice_options = LatticeSumOptions()
    lattice_options.cutoff_bohr = 15.0
    overlap = compute_overlap_lattice(basis, system, lattice_options)
    overlap_before = [np.asarray(block).copy() for block in overlap.blocks]
    density_block = np.eye(basis.nbasis)
    density = make_lattice_matrix_set(
        basis.nbasis,
        list(overlap.cells),
        [density_block],
    )

    xc_density = _density_set_gamma_or_lattice(overlap, density)

    assert xc_density is not overlap
    np.testing.assert_array_equal(xc_density.blocks[0], density_block)
    for actual, expected in zip(overlap.blocks, overlap_before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("atomic_number", "expected_periodic", "molecular_limit_tolerance"),
    [
        # Both pins moved with the #478 monopole-self-image fix, by exactly
        # the uncancelled erfc image tail +0.225 N_e^2 / L mHa:
        # He/50 bohr  -2.77186737204194  -> -2.771886044437  (1.87e-5 Ha)
        # Ar/50 bohr  -520.1895266727287 -> -520.191051001699 (1.52e-3 Ha)
        # The periodic value now equals the molecular one to ~1e-10 Ha, so
        # the pin and the molecular-limit tolerance below agree by
        # construction instead of the pin sitting inside a loose band.
        (2, -2.771886044437249, 5.0e-5),
        (18, -520.1910510016992, 2.0e-3),
    ],
)
def test_gamma_bipole_rks_atom_reaches_molecular_limit(
    atomic_number: int,
    expected_periodic: float,
    molecular_limit_tolerance: float,
):
    """A converged 50-bohr atom must not acquire the corrupt-XC J shift."""
    molecule = vq.Molecule([vq.Atom(atomic_number, [0.0, 0.0, 0.0])])
    molecular_basis = vq.BasisSet(molecule, "sto-3g")
    molecular_options = vq.RKSOptions()
    molecular_options.functional = "lda"
    molecular_options.max_iter = 80
    molecular = vq.run_rks(molecule, molecular_basis, molecular_options)

    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 50.0,
        [vq.Atom(atomic_number, [25.0, 25.0, 25.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    periodic_options = PeriodicKSOptions()
    periodic_options.functional = "lda"
    periodic_options.lattice_opts.cutoff_bohr = 15.0
    periodic_options.lattice_opts.nuclear_cutoff_bohr = 15.0
    periodic_options.max_iter = 80
    periodic_options.initial_guess = InitialGuess.HCORE
    periodic_options.use_diis = True
    periodic_options.diis_start_iter = 1

    periodic = run_pbc_bipole_rks(
        system,
        basis,
        kmesh,
        periodic_options,
        progress=False,
    )

    assert molecular.converged and periodic.converged
    assert periodic.n_iter == 2
    assert periodic.energy == pytest.approx(expected_periodic, abs=5.0e-9)
    assert periodic.energy == pytest.approx(
        molecular.energy,
        abs=molecular_limit_tolerance,
    )
    assert periodic.energy_components[-1].e_j_long_range > 0.0


# Simple-cubic Madelung constant of a unit point charge in a neutralising
# uniform background, xi_M . L. Same value pinned in
# tests/test_bipole_fock_ewald_exchange.py; the closed form of the finite-box
# monopole term is Makov & Payne, Phys. Rev. B 51, 4014 (1995), Eq. (5)
# (E = q^2 alpha_M / 2L with alpha_M = xi_M . L for the neutralised cell).
XI_TIMES_L_CUBIC = 2.837297479481


@pytest.mark.parametrize("box_bohr", [20.0, 50.0, 140.0])
def test_bipole_ewald_state_real_cutoff_converges_the_erfc_image_tail(
    box_bohr: float,
):
    """GitLab #478: the BIPOLE Ewald real cutoff must follow the pinned alpha.

    BIPOLE pins ``alpha = 2.8 / V^(1/3)``, so ``erfc(alpha . L)`` at the
    nearest image is ``erfc(2.8) = 7.5e-5`` for a cubic cell of ANY side --
    the screened kernel is scale-invariant. Sizing the Ewald real-space sum
    from ``nuclear_cutoff_bohr`` (an AO-overlap length) therefore truncates
    a term that decays only as 1/L.

    Closed-form oracle: a single unit point charge in a cubic box has Ewald
    lattice energy ``-xi_M / 2 = -XI_TIMES_L_CUBIC / (2 L)`` exactly. Before
    the fix this was wrong by ``-2.251e-4 / L`` Ha -- constant in ``err . L``
    across L = 20 .. 140, i.e. exactly the dropped image tail
    ``-(1/2) sum_{R != 0} erfc(alpha . R) / R``.
    """
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box_bohr,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
    )
    v_cell = box_bohr**3
    alpha = crystal_default_ewald_alpha(v_cell)
    tolerance = 1.0e-8

    lattice_options = LatticeSumOptions()
    lattice_options.cutoff_bohr = 15.0
    lattice_options.nuclear_cutoff_bohr = 15.0

    ewald_options = _crystal_ewald_options(
        lattice_options,
        alpha_bohr_inv=alpha,
        tolerance=tolerance,
        recip_cutoff_bohr_inv=crystal_ewald_reciprocal_cutoff(v_cell),
    )

    # The resolved envelope must screen the erfc kernel to the requested
    # tolerance -- the exact inverse of ewald.cpp's own ``auto_alpha``.
    assert math.erfc(alpha * ewald_options.real_cutoff_bohr) <= tolerance

    energy = float(ewald_nuclear_repulsion(system, ewald_options))
    exact = -0.5 * XI_TIMES_L_CUBIC / box_bohr
    # Residual scaled by L: the pre-fix defect is a CONSTANT -2.251e-4 here,
    # so a bound on err * L is what discriminates a 1/L monopole leak from
    # ordinary truncation noise (an absolute bound on err alone would pass
    # at large L for a genuine 1/L term).
    assert abs(energy - exact) * box_bohr < 1.0e-8


def test_gamma_bipole_rks_molecular_limit_has_no_monopole_self_image():
    """GitLab #478: no uncancelled ``N_e^2 / L`` term in the molecular limit.

    A neutral Gamma-only cell must reduce to the molecular result faster
    than 1/L: the electron-electron (+N_e^2), electron-nuclear (-2 N_e Z)
    and nuclear-nuclear (+Z^2) monopole self-images cancel exactly when
    ``N_e == Z_tot``. Pre-fix the electron-electron erfc sum was padded to
    the smeared-kernel range while the two nuclear channels were truncated
    at ``nuclear_cutoff_bohr``, leaving ``+0.226 N_e^2 / L`` mHa -- a
    residual that sat comfortably inside the absolute tolerances pinned
    above, which is why it went unseen.

    The gate is therefore on the *coefficient* ``dE . L / N_e^2``, not on
    ``dE``: only that quantity separates a monopole-class leak from a
    dipole/quadrupole image term (1/L^3) or SCF noise. Measured pre-fix:
    2.27e-4 Ha.bohr at both rungs. Measured post-fix: 1.7e-11, and it
    tracks ``ewald_precision`` (9.4e-8 at 1e-6, -2.2e-13 at 1e-12) instead
    of being bit-identical across it.
    """
    atomic_number = 10  # Ne: spherical, so zero dipole and zero quadrupole.
    molecule = vq.Molecule([vq.Atom(atomic_number, [0.0, 0.0, 0.0])])
    molecular_options = vq.RKSOptions()
    molecular_options.functional = "lda"
    molecular_options.max_iter = 80
    molecular = vq.run_rks(
        molecule,
        vq.BasisSet(molecule, "sto-3g"),
        molecular_options,
    )
    assert molecular.converged

    coefficients = []
    for box_bohr in (50.0, 100.0):
        system = vq.PeriodicSystem(
            3,
            np.eye(3) * box_bohr,
            [vq.Atom(atomic_number, [box_bohr / 2.0] * 3)],
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        periodic_options = PeriodicKSOptions()
        periodic_options.functional = "lda"
        periodic_options.lattice_opts.cutoff_bohr = 15.0
        periodic_options.lattice_opts.nuclear_cutoff_bohr = 15.0
        periodic_options.max_iter = 80
        periodic_options.initial_guess = InitialGuess.HCORE
        periodic = run_pbc_bipole_rks(
            system,
            basis,
            monkhorst_pack(system, [1, 1, 1]),
            periodic_options,
            progress=False,
        )
        assert periodic.converged
        # One direct cell at both rungs: the residual cannot be blamed on a
        # differing real-space cell count between the two boxes.
        assert (
            len(
                compute_overlap_lattice(
                    basis, system, periodic_options.lattice_opts
                ).cells
            )
            == 1
        )
        delta = periodic.energy - molecular.energy
        coefficients.append(abs(delta) * box_bohr / atomic_number**2)

    assert coefficients[0] < 1.0e-8, coefficients
    assert coefficients[1] < 1.0e-8, coefficients
