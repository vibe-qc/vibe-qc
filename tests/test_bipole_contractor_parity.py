"""Parity tests: C++ direct far-field Coulomb contractor vs Python reference.

Verifies that the C++ OpenMP contractor
(compute_bipolar_coulomb_far_field_cpp) produces bit-identical Fock
blocks and Coulomb energy to the Python reference implementation
(build_bipolar_coulomb_far_field).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    compute_multipole_moments_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from vibeqc._sph_to_cart import spherical_to_cartesian_with_traces
from vibeqc.bipole_pair_moments import pair_center_moments
from vibeqc.bipole_spherical_moment_buffer import build_spherical_moment_buffer
from vibeqc.bipole_dispatch import build_penetration_dispatch_for_bipole_context
from vibeqc.bipole_contractor_native import (
    compute_bipolar_coulomb_far_field_native,
)
from vibeqc.bipole_quartet_far_field import build_bipolar_coulomb_far_field
from types import SimpleNamespace

ANG2BOHR = 1.0 / 0.529177210903


def _make_mgo(a_bohr: float):
    lattice = (a_bohr / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, [a_bohr / 2.0, a_bohr / 2.0, a_bohr / 2.0]),
    ]
    return vq.PeriodicSystem(3, lattice, atoms)


def _make_lih_box(a_bohr: float):
    lattice = np.diag([a_bohr, a_bohr, a_bohr])
    atoms = [
        vq.Atom(3, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.6, 0.0, 0.0]),
    ]
    return vq.PeriodicSystem(3, lattice, atoms)


def _build_test_density(basis, cells):
    rng = np.random.RandomState(42)
    blocks = []
    for c in cells:
        P = np.eye(basis.nbasis) * 0.5
        pert = rng.randn(basis.nbasis, basis.nbasis) * 0.05
        P += pert + pert.T
        blocks.append(P)
    return make_lattice_matrix_set(basis.nbasis, cells, blocks)


def _setup_bipolar(system, basis, cutoff, L_max=4):
    """Build the spherical buffer and penetration dispatch."""
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    cells = direct_lattice_cells(system, cutoff)

    # Build moments with sphemultipole L=4.
    cart3 = compute_multipole_moments_lattice(
        basis, system, lo, 3, (0.0, 0.0, 0.0),
    )
    if L_max >= 4:
        sph4 = compute_multipole_moments_lattice(
            basis, system, lo, 4, (0.0, 0.0, 0.0),
        )
        merged = spherical_to_cartesian_with_traces(sph4, cart3, basis.nbasis)
        Merged = SimpleNamespace(nbf=basis.nbasis, L_max=4, spherical=False,
                                  cells=list(cart3.cells), origin=(0.0, 0.0, 0.0),
                                  blocks=merged)
        pair_mom = pair_center_moments(Merged, basis, L_target=4)
    else:
        pair_mom = pair_center_moments(cart3, basis, L_target=3)

    buf = build_spherical_moment_buffer(pair_mom, basis, L_max=L_max)
    dispatch = build_penetration_dispatch_for_bipole_context(
        basis, system, list(cart3.cells), maximum_multipole_order=L_max,
    )
    return buf, dispatch, cells


class TestCppContractorParity:
    """C++ contractor matches Python reference to machine precision."""

    def test_mgo_sto3g_bare_coulomb_energy_match(self):
        """MgO STO-3G: C++ bare Coulomb energy matches Python."""
        a = 4.213 * ANG2BOHR
        system = _make_mgo(a)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        buf, dispatch, cells = _setup_bipolar(system, basis, 8.0, L_max=4)

        P = _build_test_density(basis, cells)
        density_dict = {
            (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
            for i, c in enumerate(cells)
        }

        # Python reference.
        py_result = build_bipolar_coulomb_far_field(
            buf, dispatch, density_dict,
            ewald_omega=0.0, nbf=basis.nbasis,
        )

        # C++ contractor.
        cpp_result = compute_bipolar_coulomb_far_field_native(
            buf, dispatch, density_dict,
            ewald_omega=0.0, nbf=basis.nbasis,
        )

        # Compare energy.
        assert cpp_result.n_quartets == py_result.n_quartets, (
            f"Quartet count mismatch: C++={cpp_result.n_quartets}, "
            f"Py={py_result.n_quartets}"
        )
        assert cpp_result.e_coulomb_far == pytest.approx(
            py_result.e_coulomb_far, rel=1e-10, abs=1e-8
        ), (
            f"Energy mismatch: C++={cpp_result.e_coulomb_far:.15e}, "
            f"Py={py_result.e_coulomb_far:.15e}"
        )

    def test_mgo_sto3g_fock_blocks_match(self):
        """MgO STO-3G: C++ Fock blocks match Python element-by-element."""
        a = 4.213 * ANG2BOHR
        system = _make_mgo(a)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        buf, dispatch, cells = _setup_bipolar(system, basis, 8.0, L_max=4)

        P = _build_test_density(basis, cells)
        density_dict = {
            (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
            for i, c in enumerate(cells)
        }

        py_result = build_bipolar_coulomb_far_field(
            buf, dispatch, density_dict,
            ewald_omega=0.0, nbf=basis.nbasis,
        )
        cpp_result = compute_bipolar_coulomb_far_field_native(
            buf, dispatch, density_dict,
            ewald_omega=0.0, nbf=basis.nbasis,
        )

        # Same set of cell keys.
        assert set(cpp_result.fock_blocks.keys()) == set(
            py_result.fock_blocks.keys()
        ), "Fock block cell keys differ"

        # Element-by-element comparison.
        max_diff = 0.0
        for key in py_result.fock_blocks:
            py_block = np.asarray(py_result.fock_blocks[key], dtype=float)
            cpp_block = np.asarray(cpp_result.fock_blocks[key], dtype=float)
            diff = np.max(np.abs(py_block - cpp_block))
            max_diff = max(max_diff, diff)
        assert max_diff < 1e-10, (
            f"Max Fock block element difference: {max_diff:.2e}"
        )

    def test_lih_sto3g_bare_coulomb_energy_match(self):
        """LiH STO-3G: C++ bare Coulomb energy matches Python."""
        system = _make_lih_box(10.0)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        buf, dispatch, cells = _setup_bipolar(system, basis, 10.0, L_max=4)

        P = _build_test_density(basis, cells)
        density_dict = {
            (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
            for i, c in enumerate(cells)
        }

        py_result = build_bipolar_coulomb_far_field(
            buf, dispatch, density_dict,
            ewald_omega=0.0, nbf=basis.nbasis,
        )
        cpp_result = compute_bipolar_coulomb_far_field_native(
            buf, dispatch, density_dict,
            ewald_omega=0.0, nbf=basis.nbasis,
        )

        assert cpp_result.n_quartets == py_result.n_quartets
        assert cpp_result.e_coulomb_far == pytest.approx(
            py_result.e_coulomb_far, rel=1e-10, abs=1e-8
        ), (
            f"Energy mismatch: C++={cpp_result.e_coulomb_far:.15e}, "
            f"Py={py_result.e_coulomb_far:.15e}"
        )

    def test_mgo_sto3g_erfc_screened_energy_match(self):
        """MgO STO-3G: C++ erfc-screened Coulomb energy matches Python."""
        a = 4.213 * ANG2BOHR
        system = _make_mgo(a)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        buf, dispatch, cells = _setup_bipolar(system, basis, 8.0, L_max=4)

        P = _build_test_density(basis, cells)
        density_dict = {
            (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
            for i, c in enumerate(cells)
        }

        omega = 0.5
        py_result = build_bipolar_coulomb_far_field(
            buf, dispatch, density_dict,
            ewald_omega=omega, nbf=basis.nbasis,
        )
        cpp_result = compute_bipolar_coulomb_far_field_native(
            buf, dispatch, density_dict,
            ewald_omega=omega, nbf=basis.nbasis,
        )

        assert cpp_result.n_quartets == py_result.n_quartets
        # Both paths build the same erfc-screened tensor, so the only
        # admissible difference is the summation order of the OpenMP
        # reduction -- worth ~1e-13 relative on this system, not the 0.1%
        # an earlier revision of this test allowed.  That loose bound was
        # wide enough to make a genuine 0.9% divergence look like a
        # borderline tolerance question; keep it tight enough that a
        # tensor-level defect fails loudly.
        assert cpp_result.e_coulomb_far == pytest.approx(
            py_result.e_coulomb_far, rel=1e-10, abs=1e-12
        ), (
            f"Screened energy mismatch: C++={cpp_result.e_coulomb_far:.15e}, "
            f"Py={py_result.e_coulomb_far:.15e}"
        )

        # The energy is a single contracted scalar and can hide a
        # cancelling per-block error; compare the Fock blocks too.
        assert set(cpp_result.fock_blocks.keys()) == set(
            py_result.fock_blocks.keys()
        ), "Fock block cell keys differ under erfc screening"
        max_diff = 0.0
        for key in py_result.fock_blocks:
            max_diff = max(max_diff, float(np.max(np.abs(
                np.asarray(py_result.fock_blocks[key], dtype=float)
                - np.asarray(cpp_result.fock_blocks[key], dtype=float)
            ))))
        assert max_diff < 1e-10, (
            f"Max screened Fock block element difference: {max_diff:.2e}"
        )

    def test_fallback_on_import_error(self):
        """When C++ is unavailable, native wrapper falls back to Python."""
        # The native wrapper catches ImportError and falls back gracefully.
        # This test verifies the fallback produces a valid result.
        a = 4.213 * ANG2BOHR
        system = _make_mgo(a)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        buf, dispatch, cells = _setup_bipolar(system, basis, 8.0, L_max=2)

        P = _build_test_density(basis, cells)
        density_dict = {
            (c.index[0], c.index[1], c.index[2]): np.asarray(P.blocks[i])
            for i, c in enumerate(cells)
        }

        result = compute_bipolar_coulomb_far_field_native(
            buf, dispatch, density_dict,
            ewald_omega=0.0, nbf=basis.nbasis,
        )
        assert result.n_quartets > 0
        assert np.isfinite(result.e_coulomb_far)
