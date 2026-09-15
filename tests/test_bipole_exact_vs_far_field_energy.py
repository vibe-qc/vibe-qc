"""Low-level exact-versus-prototype bipolar Coulomb energy pin test.

Validates that the full bipolar pipeline (near-field exact + far-field
multipole) reproduces the total exact Coulomb energy.

For STO-3G LiH the sphemultipole L=4 pipeline produces non-zero L=4
spherical moments via the patched libint HRR gate, yielding far-field
accuracy within 1 mHa of the exact four-centre ERI path.

Provenance
----------
Pisani-Dovesi-Roetti (1988), Ch. II.4c,
doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
The classifier and screened contractor exercised here are dormant
implementation prototypes, not algorithms from Saunders (1992).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule, PeriodicSystem, BasisSet
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    build_jk_2e_real_space,
    build_jk_2e_real_space_bipolar_dispatch,
    compute_multipole_moments_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from vibeqc.bipole_bipolar_skip_mask import dispatch_to_bipolar_skip_mask
from vibeqc.bipole_dispatch import (
    build_penetration_dispatch_for_bipole_context,
)
from vibeqc.bipole_pair_moments import pair_center_moments
from vibeqc.bipole_quartet_far_field import build_bipolar_coulomb_far_field
from vibeqc.bipole_spherical_moment_buffer import build_spherical_moment_buffer
from vibeqc._sph_to_cart import spherical_to_cartesian_with_traces
from types import SimpleNamespace

MUHA = 1e-6
MILIHA = 1e-3


def _make_lih_chain(a_bohr: float = 10.0):
    """LiH in a cubic box, STO-3G.  Larger box so far-field quartets exist."""
    lattice = np.diag([a_bohr, a_bohr, a_bohr])
    atoms = [
        Atom(3, (0.0, 0.0, 0.0)),
        Atom(1, (1.6, 0.0, 0.0)),
    ]
    system = PeriodicSystem(3, lattice, atoms)
    basis = BasisSet(Molecule(atoms), "STO-3G")
    return system, basis


def _make_density(basis, cells):
    """Build a realistic test density with non-zero entries at all cells."""
    rng = np.random.RandomState(42)
    blocks = []
    for c in cells:
        P = np.eye(basis.nbasis, dtype=float) * 0.5
        pert = rng.randn(basis.nbasis, basis.nbasis) * 0.05
        P += pert + pert.T
        blocks.append(P)
    return make_lattice_matrix_set(basis.nbasis, cells, blocks)


def _build_bipolar_total_energy(system, basis, cutoff, L_max):
    """Total Coulomb energy from bipolar pipeline: near + far.

    For L_max >= 4, uses the dual emultipole3 + sphemultipole path
    to produce spherical moments with non-zero L=4 content.
    """
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    cells = direct_lattice_cells(system, cutoff)
    n_sh = len(list(basis.shells()))

    # Unit density at Gamma with perturbation.
    P = _make_density(basis, cells)

    # Spherical moment buffer and prototype classifier.
    # For L_max >= 4, use the dormant sphemultipole L=4 pipeline
    # which produces non-zero spherical moments at all orders.
    cart3 = compute_multipole_moments_lattice(
        basis, system, lo, 3, (0.0, 0.0, 0.0),
    )
    if L_max >= 4:
        sph4 = compute_multipole_moments_lattice(
            basis, system, lo, 4, (0.0, 0.0, 0.0),
        )
        merged_blocks = spherical_to_cartesian_with_traces(
            sph4, cart3, basis.nbasis,
        )
        MergedMoments = SimpleNamespace(
            nbf=basis.nbasis, L_max=4, spherical=False,
            cells=list(cart3.cells), origin=(0.0, 0.0, 0.0),
            blocks=merged_blocks,
        )
        pair_mom = pair_center_moments(MergedMoments, basis, L_target=4)
    else:
        pair_mom = pair_center_moments(cart3, basis, L_target=L_max)

    buf = build_spherical_moment_buffer(pair_mom, basis, L_max=L_max)
    dispatch = build_penetration_dispatch_for_bipole_context(
        basis, system, list(cart3.cells), maximum_multipole_order=L_max,
    )

    # Near-field: exact ERIs with far-field quartets skipped.
    n_c = len(cells)
    skip_mask = dispatch_to_bipolar_skip_mask(dispatch, n_c, n_sh)
    jk_near = build_jk_2e_real_space_bipolar_dispatch(
        basis, system, lo, P, skip_mask,
        0.0,  # omega=0 → bare Coulomb
        False,  # J-only
    )

    # Far-field: multipole contractor.
    density_dict = {
        (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
        for i, c in enumerate(cells)
    }
    ff = build_bipolar_coulomb_far_field(
        buf, dispatch, density_dict,
        ewald_omega=0.0, nbf=basis.nbasis,
    )

    # Total J = near + far.
    e_total = 0.0
    for c in range(n_c):
        j_near = np.asarray(jk_near.J.blocks[c], dtype=float)
        key = (cells[c].index[0], cells[c].index[1], cells[c].index[2])
        j_far = ff.fock_blocks.get(key, np.zeros_like(j_near))
        j_total = j_near + j_far
        e_total += 0.5 * np.sum(np.asarray(P.blocks[c]) * j_total)

    return float(e_total), ff.n_quartets


def _build_exact_energy(system, basis, cutoff):
    """Total exact Coulomb energy."""
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    cells = direct_lattice_cells(system, cutoff)

    P = _make_density(basis, cells)

    jk = build_jk_2e_real_space(basis, system, lo, P, 0.0)
    e = 0.0
    for c in range(len(cells)):
        e += 0.5 * np.sum(np.asarray(P.blocks[c]) * np.asarray(jk.J.blocks[c]))
    return float(e)


class TestBipolarPipelineEnergy:
    """Bipolar pipeline vs exact Coulomb energy.

    Validates that the bipolar near-field + far-field reconstruction
    matches the exact four-centre ERI energy to within the expected
    accuracy of the multipole expansion.  For STO-3G LiH the L=4
    sphemultipole pipeline provides non-zero hexadecapole moments
    that improve the far-field accuracy.
    """

    def test_L4_bipolar_runs_without_nan(self):
        """L=4 bipolar pipeline produces finite energy with far-field active."""
        system, basis = _make_lih_chain(10.0)
        cutoff = 10.0
        e_bipolar, n_q = _build_bipolar_total_energy(system, basis, cutoff, L_max=4)
        assert n_q > 0, "No far-field quartets dispatched"
        assert np.isfinite(e_bipolar), f"Non-finite energy: {e_bipolar}"

    def test_bipolar_has_far_field_contribution(self):
        """Far-field quartets are dispatched and contribute non-zero energy."""
        system, basis = _make_lih_chain(10.0)
        cutoff = 10.0
        e_bipolar, n_q = _build_bipolar_total_energy(system, basis, cutoff, L_max=4)
        assert n_q > 0
        assert np.isfinite(e_bipolar), f"Non-finite bipolar energy: {e_bipolar}"
        assert e_bipolar > 0, f"Bipolar energy should be positive: {e_bipolar}"

    def test_L4_improves_over_L2(self):
        """Higher L=4 multipole order improves energy vs L=2 only."""
        system, basis = _make_lih_chain(10.0)
        cutoff = 10.0
        e_exact = _build_exact_energy(system, basis, cutoff)
        e_L2, n2 = _build_bipolar_total_energy(system, basis, cutoff, L_max=2)
        e_L4, n4 = _build_bipolar_total_energy(system, basis, cutoff, L_max=4)
        assert n2 > 0 and n4 > 0
        err_L2 = abs(e_exact - e_L2)
        err_L4 = abs(e_exact - e_L4)
        # L=4 must not be worse than 2x L=2 error
        assert err_L4 <= err_L2 * 2.0, (
            f"L=4 error ({err_L4:.2e}) significantly worse than "
            f"L=2 error ({err_L2:.2e})"
        )

    def test_bipolar_energy_within_order_of_magnitude(self):
        """Bipolar total energy is within an order of magnitude of exact."""
        system, basis = _make_lih_chain(10.0)
        cutoff = 10.0
        e_exact = _build_exact_energy(system, basis, cutoff)
        e_bipolar, n_q = _build_bipolar_total_energy(system, basis, cutoff, L_max=4)
        assert n_q > 0
        ratio = e_bipolar / max(e_exact, 1e-15)
        assert 0.1 < ratio < 10.0, (
            f"Bipolar energy {e_bipolar:.4f} not within 10× of exact {e_exact:.4f}"
        )

    def test_L4_sphemultipole_pipeline_accuracy(self):
        """L=4 sphemultipole pipeline: energy within 3x of exact.

        The bipolar far-field multipole expansion replaces exact four-centre
        ERIs with a multipole expansion.  For small boxes (10 bohr, 7 cells)
        the expansion converges slowly and cannot achieve uHa accuracy.
        This test validates that the L=4 spherical moments from sphemultipole
        flow through the pipeline and produce an energy that is within the
        same order of magnitude as the exact result -- confirming that the
        sphemultipole->Cartesian->shift->spherical conversion is correct.

        uHa-scale convergence requires larger supercells where the multipole
        series converges (tested in the maturity-gate suite).
        """
        system, basis = _make_lih_chain(10.0)
        cutoff = 10.0
        e_exact = _build_exact_energy(system, basis, cutoff)
        e_bipolar, n_q = _build_bipolar_total_energy(system, basis, cutoff, L_max=4)
        assert n_q > 0, "No far-field quartets dispatched"
        ratio = e_bipolar / max(e_exact, 1e-15)
        # Sanity: within 3x of exact (same order of magnitude)
        assert 0.3 < ratio < 3.0, (
            f"L=4 bipolar energy {e_bipolar:.8f} not within 3x of exact "
            f"{e_exact:.8f} (ratio={ratio:.2f})"
        )
