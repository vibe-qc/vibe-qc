"""Toroidal megacell periodic Γ RI-MP2 validation.

The tests cover internal contraction parity, finite-torus routing, structural
invariants, local-correlation reductions, and thermodynamic-limit behaviour.
Exact character-mesh/KMP2 parity is tested separately on a matched finite
Hamiltonian in ``test_periodic_aiccm2026dev_b_posthf.py``.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_toroidal_mp2 import (
    ToroidalMP2Result,
    ToroidalPairClassification,
    ToroidalPairResult,
    ToroidalTDLResult,
    _assign_wannier_cells,
    _prepare_toroidal_image_search,
    _toroidal_minimum_image,
    build_toroidal_supercell,
    classify_toroidal_pairs,
    toroidal_mp2_gamma,
    toroidal_mp2_gamma_pairs,
    toroidal_mp2_tdl,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _h2_chain_cell(
    spacing_bohr: float = 6.0, hh_bond: float = 1.4, transverse: float = 20.0
) -> vq.PeriodicSystem:
    """One H2 per cell; periodic along z; transverse vacuum padding.

    Constructed as a 3D-periodic system with vacuum padding in x and y so
    the chain is effectively 1D.  For the toroidal route, the supercell
    re-uses the same transverse box, so transverse chain-chain coupling is
    controlled by the vacuum size.
    """
    return vq.PeriodicSystem(
        3,
        np.diag([transverse, transverse, spacing_bohr]),
        [
            vq.Atom(
                1, [transverse / 2, transverse / 2, spacing_bohr / 2 - hh_bond / 2]
            ),
            vq.Atom(
                1, [transverse / 2, transverse / 2, spacing_bohr / 2 + hh_bond / 2]
            ),
        ],
    )


def _skew_helium_cell() -> tuple[vq.PeriodicSystem, np.ndarray, np.ndarray]:
    """One He atom in a deliberately skew column-vector lattice."""
    lattice = np.array(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    frac = np.array([0.17, 0.29, 0.41])
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, lattice @ frac)])
    return system, lattice, frac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_toroidal_mp2_sanity():
    """Basic invariants of the toroidal MP2 route."""
    system = _h2_chain_cell()

    # (1,1,1) = unit cell → periodic Γ DF-MP2 on one cell.
    r = toroidal_mp2_gamma(system, "sto-3g", (1, 1, 1))
    assert isinstance(r, ToroidalMP2Result)
    assert r.nrep == (1, 1, 1)
    assert r.n_cells == 1
    assert r.n_atoms == 2
    assert r.n_occ == 1  # H2/STO-3G: 2 e-, 1 occupied MO
    assert r.n_vir == r.n_bf - 1
    assert r.n_fit > 0
    # Correlation energy is negative (physical).
    assert r.e_corr < 0
    assert r.e_corr_per_cell < 0
    assert r.e_corr_per_cell == pytest.approx(r.e_corr, abs=1e-14)
    # Total energy is HF + corr.
    assert r.e_total == pytest.approx(r.e_hf + r.e_corr, abs=1e-14)

    # (1,1,2) = 2-cell supercell along z — per-cell converges to a finite limit.
    r2 = toroidal_mp2_gamma(system, "sto-3g", (1, 1, 2))
    assert r2.n_cells == 2
    assert r2.n_atoms == 4
    assert r2.e_corr < r.e_corr  # total corr more negative with 2x electrons
    assert r2.e_corr_per_cell < 0
    # Per-cell values should be on the same order.
    assert abs(r2.e_corr_per_cell - r.e_corr_per_cell) < 1e-2


@pytest.mark.parametrize(
    "route_name",
    ("canonical", "pairs", "dlpno-mp2", "dlpno-ccsd", "dlpno-ccsd(t)"),
)
def test_toroidal_routes_run_bvk_gamma_hf(monkeypatch, route_name):
    """Every toroidal post-HF API runs its HF on the GDF Hamiltonian at Gamma.

    Until #566 they forced the Ewald-J / molecular-limit-K branch of
    ``run_rhf_periodic_gamma_gdf``, whose exchange is not invariant under a
    translation of the torus.
    """
    import vibeqc.periodic_rhf_gdf as rhf_gdf
    from vibeqc.periodic_toroidal_mp2 import (
        toroidal_dlpno_ccsd,
        toroidal_dlpno_ccsd_t,
        toroidal_dlpno_mp2,
    )

    routes = {
        "canonical": toroidal_mp2_gamma,
        "pairs": toroidal_mp2_gamma_pairs,
        "dlpno-mp2": toroidal_dlpno_mp2,
        "dlpno-ccsd": toroidal_dlpno_ccsd,
        "dlpno-ccsd(t)": toroidal_dlpno_ccsd_t,
    }
    captured = []

    class _StopAfterRouting(Exception):
        pass

    def _capture_bulk_call(*args, **kwargs):
        captured.append((args, kwargs))
        raise _StopAfterRouting

    def _refuse_legacy_route(*args, **kwargs):
        raise AssertionError("toroidal HF must not use the legacy Gamma driver")

    monkeypatch.setattr(rhf_gdf, "_bulk_gamma_k_gdf", _capture_bulk_call)
    monkeypatch.setattr(rhf_gdf, "run_rhf_periodic_gamma_gdf", _refuse_legacy_route)
    options = vq.PeriodicRHFOptions()
    options.max_iter = 77

    with pytest.raises(_StopAfterRouting):
        routes[route_name](_h2_chain_cell(), "sto-3g", (1, 1, 3), rhf_options=options)

    assert len(captured) == 1
    args, kwargs = captured[0]
    assert len(args[0].unit_cell) == 6
    assert args[2] is options
    assert kwargs["aux_basis"] == "def2-svp-jk"


@pytest.mark.parametrize("n_cells", (3, 4))
def test_toroidal_hf_is_the_unit_cell_kmesh_hf(n_cells):
    """The supercell HF is the unit-cell KRHF on the N-point mesh (BvK).

    Its density commutes with the one-cell translation of the torus. The
    molecular-limit exchange used before #566 broke both: K changed by
    3.2e-3 (N = 3) and 4.4e-3 (N = 4) under the translation, and the
    converged density by 5.5e-4 and 2.5e-4.
    """
    from vibeqc.periodic_toroidal_mp2 import _toroidal_bvk_hf

    unit = _h2_chain_cell()
    supercell = build_toroidal_supercell(unit, (1, 1, n_cells))
    basis = vq.BasisSet(supercell.unit_cell_molecule(), "sto-3g")
    hf = _toroidal_bvk_hf(supercell, basis, None, "def2-svp-jk")
    assert hf.converged
    assert hf.exchange_q0 == "BvK-ewald"
    assert not np.iscomplexobj(hf.mo_coeffs)

    shift = (np.arange(basis.nbasis) + basis.nbasis // n_cells) % basis.nbasis
    density = np.asarray(hf.density)
    translated = np.empty_like(density)
    translated[np.ix_(shift, shift)] = density
    assert np.abs(translated - density).max() < 1e-9

    kmesh = vq.run_krhf_periodic_gdf(
        unit, vq.BasisSet(unit.unit_cell_molecule(), "sto-3g"),
        (1, 1, n_cells), aux_basis="def2-svp-jk", progress=False,
    )
    assert hf.energy / n_cells == pytest.approx(kmesh.energy, abs=1e-10)


def test_toroidal_mp2_nrep3_retains_periodic_hf_beyond_cutoff():
    """Crossing the lattice cutoff must not switch the torus to molecular HF."""
    system = _h2_chain_cell()
    options = vq.PeriodicRHFOptions()
    supercell_period = 3.0 * 6.0
    assert options.lattice_opts.cutoff_bohr < supercell_period

    result = toroidal_mp2_gamma(
        system,
        "sto-3g",
        (1, 1, 3),
        rhf_options=options,
    )

    # Deterministic single-thread repeats are bit-identical.
    #
    # The HF is the GDF Hamiltonian at Gamma with exxdiv='ewald' exchange,
    # the unit-cell KRHF on the 3-point mesh (#566;
    # test_toroidal_hf_is_the_unit_cell_kmesh_hf). PySCF 2.14 KRHF/FFTDF at
    # kmesh (1, 1, 3) gives -1.1178240341 Ha/cell, 2.3e-5 above the pin; the
    # same offset at 2 and 4 cells is the def2-svp-jk fitting error.
    #
    # Pin history. The cutoff-selected molecular branch returned
    # -2.7879828251544 / -0.0049169556998 Ha/cell (#166/#183). The correlation
    # pin was -0.0135092212660 until the compcell DF tensor summed its
    # auxiliary images (23d5e8db, #172) and its AFT part took in the seam AO
    # pairs (0ef4b2fd, #438). The SCF pins were -1.1164515356204 and
    # -0.0132120104972 until the Ewald-J HF reached the AO pairs across the
    # seam (8c24e3af), then -1.1164536588821 and -0.0131980528416 until #566
    # replaced its molecular-limit exchange, which was not invariant under a
    # translation of the torus (K residual 3.2e-3 at this size).
    assert result.e_hf_per_cell == pytest.approx(-1.1178468400477, abs=2e-8)
    assert result.e_corr_per_cell == pytest.approx(-0.0131780803597, abs=2e-8)
    # The caller's options are untouched.
    assert options.lattice_opts.cutoff_bohr == 15.0


def test_toroidal_mp2_cpp_matches_python_backend():
    """C++ MP2 energy kernel == retained Python reference (port parity check).

    toroidal_mp2_gamma defaults to the C++/OpenMP contraction
    (aiccm2026dev_b_mp2_energy_from_lov reduced to n_k=1); the private
    ``_energy_backend="python"`` selector runs the reference loop over the
    explicit MO-ERI matrix. The two must agree to ~1e-10 Ha (the cross-path
    floating-point-reassociation floor — a real porting bug would shift the
    correlation energy by mHa).
    """
    system = _h2_chain_cell()
    for nrep in [(1, 1, 1), (1, 1, 2), (1, 1, 3)]:
        r_cpp = toroidal_mp2_gamma(system, "sto-3g", nrep, _energy_backend="cpp")
        r_py = toroidal_mp2_gamma(system, "sto-3g", nrep, _energy_backend="python")
        assert r_cpp.e_corr == pytest.approx(r_py.e_corr, abs=1e-10), (
            f"cpp vs python MP2 mismatch at nrep={nrep}: "
            f"cpp={r_cpp.e_corr:.12f}, python={r_py.e_corr:.12f}"
        )
        assert r_cpp.e_total == pytest.approx(r_py.e_total, abs=1e-10)


def test_build_toroidal_supercell_structure():
    """build_toroidal_supercell produces correctly replicated atoms + lattice."""
    system = _h2_chain_cell(spacing_bohr=6.0)

    # (1,1,2) supercell along z.
    sc = build_toroidal_supercell(system, (1, 1, 2))
    lattice_sc = np.asarray(sc.lattice, dtype=float)
    np.testing.assert_allclose(
        lattice_sc[0], [20.0, 0.0, 0.0], atol=1e-12, err_msg="x lattice unchanged"
    )
    np.testing.assert_allclose(
        lattice_sc[1], [0.0, 20.0, 0.0], atol=1e-12, err_msg="y lattice unchanged"
    )
    np.testing.assert_allclose(
        lattice_sc[2], [0.0, 0.0, 12.0], atol=1e-12, err_msg="z lattice doubled"
    )

    atoms = list(sc.unit_cell)
    assert len(atoms) == 4  # 2 cells × 2 H each
    # All atoms are within the super-lattice (fractional coords in [0,1)).
    for a in atoms:
        xyz = np.asarray(a.xyz, dtype=float)
        frac = np.linalg.solve(lattice_sc, xyz)
        assert np.all((frac >= -1e-12) & (frac < 1.0 + 1e-12)), (
            f"atom at {xyz} has fractional coords {frac} outside [0,1)"
        )


def test_build_toroidal_supercell_skew_lattice_uses_column_vectors():
    """Replication follows PeriodicSystem's column-vector lattice contract."""
    system, lattice, atom_frac = _skew_helium_cell()
    nrep = np.array([2, 3, 1], dtype=int)

    sc = build_toroidal_supercell(system, tuple(nrep))
    lattice_sc = np.asarray(sc.lattice, dtype=float)
    np.testing.assert_allclose(lattice_sc, lattice @ np.diag(nrep), atol=1e-12)

    actual_frac = np.array(
        [np.linalg.solve(lattice_sc, np.asarray(atom.xyz)) for atom in sc.unit_cell]
    )
    expected_frac = np.array(
        [
            (atom_frac + np.array(index, dtype=float)) / nrep
            for index in product(*(range(int(n)) for n in nrep))
        ]
    )
    actual_frac = np.array(sorted(map(tuple, np.round(actual_frac, 12))))
    expected_frac = np.array(sorted(map(tuple, np.round(expected_frac, 12))))
    np.testing.assert_allclose(actual_frac, expected_frac, atol=1e-12)


def test_assign_wannier_cells_skew_lattice_uses_column_vectors():
    """Cartesian centroids map back through ``A f = r`` on a skew cell."""
    _system, lattice, _atom_frac = _skew_helium_cell()
    nrep = (2, 3, 2)
    fractional = np.array([[0.001, 0.001, 0.95], [1.99, 0.73, 0.26]])
    centroids = np.array([lattice @ frac for frac in fractional], dtype=float)
    tiles = np.floor(fractional).astype(int)
    np.testing.assert_array_equal(
        _assign_wannier_cells(centroids, lattice, nrep), tiles
    )


def test_toroidal_supercell_habc_same_as_unit_cell():
    """The (1,1,1) toroidal supercell is equivalent to the unit cell MP2."""
    system = _h2_chain_cell()
    r = toroidal_mp2_gamma(system, "sto-3g", (1, 1, 1))

    sc = build_toroidal_supercell(system, (1, 1, 1))
    r2 = toroidal_mp2_gamma(sc, "sto-3g", (1, 1, 1))
    # On the (1,1,1) supercell of the (1,1,1) cell, we should get the
    # same per-cell result.
    assert r2.e_corr_per_cell == pytest.approx(r.e_corr_per_cell, abs=1e-10)


# ---------------------------------------------------------------------------
# Stage 5b — Translational pair-family decomposition
# ---------------------------------------------------------------------------


def test_pair_decomposition_matches_all_pairs():
    """5b pair-grouped total agrees with 5a to the pinned tolerance."""
    from vibeqc.periodic_toroidal_mp2 import (
        ToroidalPairResult,
        toroidal_mp2_gamma_pairs,
    )

    system = _h2_chain_cell()
    for nrep in [(1, 1, 2), (1, 1, 3)]:
        r5a = toroidal_mp2_gamma(system, "sto-3g", nrep)
        r5b = toroidal_mp2_gamma_pairs(system, "sto-3g", nrep)

        assert isinstance(r5b, ToroidalPairResult)
        assert r5b.e_corr_total == pytest.approx(r5a.e_corr, abs=1e-14), (
            f"5b total != 5a all-pairs at nrep={nrep}: "
            f"{r5b.e_corr_total:.15f} vs {r5a.e_corr:.15f}"
        )
        assert r5b.e_corr_per_cell == pytest.approx(r5a.e_corr_per_cell, abs=1e-14)
        # sum(e_corr_by_L) must match e_corr_total internally.
        assert sum(r5b.e_corr_by_L.values()) == pytest.approx(
            r5b.e_corr_total, abs=1e-14
        )


def test_pair_L_wannier_cell_assignment():
    """Wannier cell assignment: indices in [0,n), total pairs = n_occ^2."""
    from vibeqc.periodic_toroidal_mp2 import toroidal_mp2_gamma_pairs

    system = _h2_chain_cell()
    for n in (2, 3, 4):
        r5b = toroidal_mp2_gamma_pairs(system, "sto-3g", (1, 1, n))
        cells = r5b.wannier_cells
        n_occ = int(r5b.n_occ)
        assert n_occ == n
        assert np.all(cells[:, 0] == 0)
        assert np.all(cells[:, 1] == 0)
        # Cell indices must be in [0, n).
        assert np.all(cells[:, 2] >= 0) and np.all(cells[:, 2] < n)
        # Total pair count across all weights = n_occ^2.
        total_pairs = sum(r5b.num_pairs_by_L.values())
        assert total_pairs == n_occ * n_occ
        # Unique L = n (one per relative z-shift on the torus).
        assert r5b.unique_L == n


@pytest.fixture(scope="module")
def _pairs_n4():
    from vibeqc.periodic_toroidal_mp2 import toroidal_mp2_gamma_pairs

    return toroidal_mp2_gamma_pairs(_h2_chain_cell(), "sto-3g", (1, 1, 4))


def test_pair_correlation_wrap_around_families_agree(_pairs_n4):
    """Per-L partition: the L and N - L families agree and the parts sum up."""
    by_L = _pairs_n4.e_corr_by_L
    assert set(by_L) == {(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)}
    # Swapping i and j maps L to N - L, so these agree by construction.
    assert by_L[(0, 0, 1)] == pytest.approx(by_L[(0, 0, 3)], abs=1e-12)
    assert sum(by_L.values()) == pytest.approx(_pairs_n4.e_corr_total, abs=1e-14)


def test_pair_correlation_decays_with_distance(_pairs_n4):
    """Per-L MP2 correlation is largest in-cell and decays with |L|.

    The split is over Wannier pairs (#566): about -5.2e-2, -2.5e-4 and
    -1.9e-5 Ha for |L| = 0, 1 and 2 on the 4-cell chain. Assigning the
    canonical (Bloch) orbitals of the torus to cells left this decided by
    noise, and the test was a non-strict xfail.
    """
    by_L = _pairs_n4.e_corr_by_L
    for dL in by_L:
        if dL != (0, 0, 0):
            assert by_L[dL] > by_L[(0, 0, 0)]  # less negative = smaller magnitude
    assert abs(by_L[(0, 0, 2)]) < abs(by_L[(0, 0, 1)])


# ---------------------------------------------------------------------------
# Stage 5c — Pair screening (dipole-approximation classifier)
# ---------------------------------------------------------------------------


def test_pair_classifier_basics():
    """Pair classifier produces correct counts and sorts by distance."""
    system = _h2_chain_cell()
    cl = classify_toroidal_pairs(system, (1, 1, 8))

    assert isinstance(cl, ToroidalPairClassification)
    assert cl.n_cells == 8
    # (0,0,0) is always strong (same-cell).
    assert (0, 0, 0) in cl.strong_L
    assert (0, 0, 0) not in cl.distant_L
    # Total L families = n_cells.
    assert cl.n_strong + cl.n_weak + cl.n_distant == cl.n_cells
    # Distances increase only up to the halfway point, then decrease by the
    # inverse-image symmetry L <-> -L on the torus.
    for dz in range(1, 4):
        assert cl.dist_by_L[(0, 0, dz)] < cl.dist_by_L[(0, 0, dz + 1)]
    for dz in range(1, 8):
        inverse = (0, 0, (-dz) % 8)
        assert cl.dist_by_L[(0, 0, dz)] == pytest.approx(
            cl.dist_by_L[inverse], abs=1e-12
        )
        assert cl.e_est_by_L[(0, 0, dz)] == pytest.approx(
            cl.e_est_by_L[inverse], abs=1e-14
        )


def test_pair_classifier_skew_lattice_matches_minimum_image_oracle():
    """Pair distances use exact closest images for a skew unequal torus."""
    system, lattice, _atom_frac = _skew_helium_cell()
    nrep = np.array([2, 3, 2], dtype=int)
    cl = classify_toroidal_pairs(
        system,
        tuple(nrep),
        r_close=0.0,
        r_cutoff=1.0e9,
        tcut_pairs=0.0,
    )
    lattice_sc = lattice @ np.diag(nrep)

    for L in product(*(range(int(n)) for n in nrep)):
        raw = lattice @ np.asarray(L, dtype=float)
        expected = min(
            np.linalg.norm(raw - lattice_sc @ np.asarray(image, dtype=float))
            for image in product(range(-2, 3), repeat=3)
        )
        assert cl.dist_by_L[L] == pytest.approx(expected, abs=1e-12)
        inverse = tuple(((-np.asarray(L, dtype=int)) % nrep).tolist())
        assert cl.dist_by_L[L] == pytest.approx(
            cl.dist_by_L[inverse], abs=1e-12
        )


def test_pair_classifier_acute_lattice_uses_bounded_exact_search():
    """An unreduced acute basis has a short non-componentwise torus image."""
    lattice = np.array(
        [
            [10.0, 15.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 8.0],
        ]
    )
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    nrep = np.array([5, 5, 1], dtype=int)
    cl = classify_toroidal_pairs(
        system,
        tuple(nrep),
        r_close=0.0,
        r_cutoff=1.0e9,
        tcut_pairs=0.0,
    )

    # For L=(2,2,0), independent fractional wrapping keeps the long raw
    # vector A@(2,2,0).  The exact image m=(1,0,0) is A@(-3,2,0)=(0,1,0).
    assert cl.dist_by_L[(2, 2, 0)] == pytest.approx(1.0, abs=1e-12)


def test_toroidal_minimum_image_preserves_large_integer_shear():
    """LLL coordinates avoid cancellation in a severely unreduced basis."""
    super_lattice = np.array(
        [
            [1.0, 1.0e12, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    search = _prepare_toroidal_image_search(super_lattice)
    displacement = _toroidal_minimum_image(
        (49, 49, 0), (100, 100, 1), search
    )
    np.testing.assert_allclose(displacement, [0.49, 0.49, 0.0], atol=1e-14)


def test_pair_classifier_thin_skew_cell_reduces_before_search():
    """A valid thin skew cell avoids the unreduced decoder performance cliff."""
    lattice = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 0.0, 1.0e-5],
        ]
    )
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    cl = classify_toroidal_pairs(
        system, (1, 1, 2), r_close=0.0, r_cutoff=1.0e9, tcut_pairs=0.0
    )
    expected = np.sqrt(0.5 + 1.0e-10)
    assert cl.dist_by_L[(0, 0, 1)] == pytest.approx(expected, abs=1e-12)


def test_pair_classifier_rejects_numerically_singular_lattice():
    """An intrinsically collapsed cell fails before CVP enumeration."""
    lattice = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0e-13, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    with pytest.raises(ValueError, match="numerically full-rank lattice"):
        classify_toroidal_pairs(system, (5, 5, 1))


def test_pair_classifier_rejects_nonpositive_replication():
    """Minimum-image classification requires a nonempty torus."""
    with pytest.raises(ValueError, match="nrep components must be >= 1"):
        classify_toroidal_pairs(_h2_chain_cell(), (1, 0, 2))


@pytest.mark.parametrize(
    ("r_close", "r_cutoff"),
    [(-1.0, 15.0), (8.0, 8.0), (9.0, 8.0)],
)
def test_pair_classifier_rejects_ambiguous_distance_thresholds(
    r_close: float, r_cutoff: float
):
    """The always-strong guard must end strictly before the hard cutoff."""
    with pytest.raises(ValueError, match="0 <= r_close < r_cutoff"):
        classify_toroidal_pairs(
            _h2_chain_cell(), (1, 1, 2), r_close=r_close, r_cutoff=r_cutoff
        )


def test_pair_classifier_r_close_default():
    """Default r_close=8.0 bohr captures nearest-neighbour chain cells."""
    system = _h2_chain_cell()
    # 6-bohr spacing → nearest-neighbour at 6 bohr ≤ 8 → strong.
    cl = classify_toroidal_pairs(system, (1, 1, 8))
    assert (0, 0, 1) in cl.strong_L
    # Second neighbour at 12 bohr > 8 → weak or distant.
    assert (0, 0, 2) not in cl.strong_L


def test_pair_classifier_all_strong_at_zero_pair_threshold():
    """With effectively unbounded r_close and tcut_pairs=0, all are strong."""
    system = _h2_chain_cell()
    cl = classify_toroidal_pairs(
        system, (1, 1, 4), r_close=1e9, r_cutoff=2e9, tcut_pairs=0
    )
    assert cl.n_strong == cl.n_cells
    assert cl.n_weak == 0
    assert cl.n_distant == 0


def test_pair_classifier_all_distant_at_tight_r_close():
    """With r_close=0, only L=0 is strong (same-cell)."""
    system = _h2_chain_cell()
    cl = classify_toroidal_pairs(system, (1, 1, 4), r_close=0.0)
    assert cl.n_strong == 1  # only (0,0,0)
    assert (0, 0, 0) in cl.strong_L
    # All others are weak or distant.
    assert cl.n_distant + cl.n_weak == cl.n_cells - 1
    assert cl.total_distant_cell_pairs == cl.n_distant * cl.n_cells
    # Historical API name is retained, but it counts cell-pair placements.
    assert cl.total_pairs_skipped == cl.total_distant_cell_pairs


# ---------------------------------------------------------------------------
# Toroidal TDL extrapolation
# ---------------------------------------------------------------------------


def test_toroidal_tdl_basics():
    """Toroidal TDL produces a sensible bulk estimate from a size series."""
    system = _h2_chain_cell()
    tdl = toroidal_mp2_tdl(system, "sto-3g", sizes=(2, 3, 4, 5), axis=2)

    assert isinstance(tdl, ToroidalTDLResult)
    assert tdl.sizes == (2, 3, 4, 5)
    assert len(tdl.e_corr_total) == 4
    assert len(tdl.e_corr_per_cell) == 4
    assert len(tdl.increments) == 3
    # Bulk per-cell energy must be negative and finite.
    assert tdl.e_corr_per_cell_bulk < 0
    assert abs(tdl.e_corr_per_cell_bulk) < 1.0
    # Fit residual should be small for a clean 1-D chain series.
    # The toroidal (BvK) finite-size error is larger than the open-megacell
    # one because periodic Ewald energies converge slower than open-boundary.
    assert tdl.fit_max_residual < 1e-4
    # The finite-N per-cell averages converge toward the bulk from above:
    # they become more negative with N (-0.0130227, -0.0131781, -0.0132412,
    # -0.0132738 Ha/cell for N = 2..5 on the BvK HF of #566).  On the
    # molecular-limit-exchange HF before #566, PySCF FFTDF integrals on the
    # same orbitals ordered the same way (-0.0130839, -0.0132097, -0.0132623,
    # -0.0132898 for N = 2..5).  The opposite order asserted before
    # #172 (23d5e8db) came from the single-image DF tensor, whose seam error
    # (-5.2e-4 Ha/cell at N = 2, -1.4e-4 at N = 5) shrinks with N.
    for i in range(len(tdl.e_corr_per_cell) - 1):
        assert tdl.e_corr_per_cell[i] > tdl.e_corr_per_cell[i + 1]


def test_toroidal_tdl_increments():
    """Increments converge toward the bulk value from below."""
    system = _h2_chain_cell()
    tdl = toroidal_mp2_tdl(system, "sto-3g", sizes=(2, 3, 4, 5, 6), axis=2)
    # Increments = E_corr(N) - E_corr(N-1), which approach the bulk value.
    assert len(tdl.increments) == 4
    bulk = tdl.e_corr_per_cell_bulk
    # The increments should be within ~1 mHa of the fit slope.
    for inc in tdl.increments:
        assert inc == pytest.approx(bulk, abs=1e-3)


# ---------------------------------------------------------------------------
# Stage 5c — DLPNO-MP2 on the toroidal megacell (tests from previous session)


# ---------------------------------------------------------------------------
# Stage 5c — DLPNO local approximation on the toroidal megacell
# ---------------------------------------------------------------------------


def test_toroidal_local_defaults_pin_pre_sweep_periodic_convention():
    """Molecular #140/#448 defaults cannot repin toroidal evidence."""
    from vibeqc.periodic_toroidal_mp2 import (
        _pre_sweep_toroidal_dlpno_cc_options,
        _pre_sweep_toroidal_dlpno_mp2_options,
    )

    mp2 = _pre_sweep_toroidal_dlpno_mp2_options()
    assert (
        mp2.n_frozen,
        mp2.tcut_pno,
        mp2.tcut_pno_weak,
        mp2.tcut_mkn,
        mp2.tcut_pairs,
        mp2.tcut_pairs_weak,
        mp2.pno_norm,
        mp2.domain_criterion,
    ) == (0, 1e-8, 1e-7, 1e-3, 1e-6, 1e-4, "legacy", "mulliken")

    cc = _pre_sweep_toroidal_dlpno_cc_options()
    assert (
        cc.n_frozen,
        cc.tcut_pno,
        cc.tcut_mkn,
        cc.tcut_pairs,
        cc.residual_domain,
        cc.pno_norm,
        cc.pno_correction,
        cc.tcut_pno_diag_scale,
    ) == (
        0,
        1e-7,
        0.0,
        1e-4,
        "pair",
        "legacy",
        False,
        1.0,
    )


def test_toroidal_dlpno_matches_canonical_exactness():
    """DLPNO-MP2 in the exactness limit == canonical toroidal MP2.

    With full domains (tcut_mkn=0), no PNO truncation (tcut_pno=0), and no
    pair screening (tcut_pairs=0), the DLPNO-MP2 must reproduce the canonical
    RI-MP2 to machine precision — the M2 exactness gate, now on the toroidal
    megacell.
    """
    from vibeqc.dlpno.mp2 import DLPNOMP2Options
    from vibeqc.periodic_toroidal_mp2 import toroidal_dlpno_mp2

    system = _h2_chain_cell()
    exact = DLPNOMP2Options(
        n_frozen=0,
        tcut_pno=0,
        tcut_pno_weak=0,
        tcut_mkn=0,
        tcut_pairs=0,
        tcut_pairs_weak=0,
    )
    for nk in (2, 3):
        r5a = toroidal_mp2_gamma(system, "sto-3g", (1, 1, nk))
        r5c = toroidal_dlpno_mp2(system, "sto-3g", (1, 1, nk), dlpno_options=exact)
        assert r5c.dlpno_result.converged
        assert r5c.e_corr_per_cell == pytest.approx(r5a.e_corr_per_cell, abs=1e-8)


def test_toroidal_dlpno_defaults_converge():
    """The historical periodic DLPNO-MP2 default stays all-electron."""
    from vibeqc.periodic_toroidal_mp2 import toroidal_dlpno_mp2

    system = _h2_chain_cell()
    r = toroidal_dlpno_mp2(system, "sto-3g", (1, 1, 2))
    assert r.dlpno_result.converged
    assert r.dlpno_result.n_frozen == 0
    assert r.dlpno_result.n_iter <= 20
    assert r.e_corr < 0
    assert r.e_corr_per_cell < 0
    # Default DLPNO should be within ~1 mHa of canonical at this size.
    r5a = toroidal_mp2_gamma(system, "sto-3g", (1, 1, 2))
    assert abs(r.e_corr_per_cell - r5a.e_corr_per_cell) < 1e-3


def test_toroidal_dlpno_energy_decomposition():
    """DLPNO-MP2 result carries physically sensible energy components."""
    from vibeqc.periodic_toroidal_mp2 import toroidal_dlpno_mp2

    system = _h2_chain_cell()
    r = toroidal_dlpno_mp2(system, "sto-3g", (1, 1, 2))
    dlpno = r.dlpno_result
    # Iterated energy dominates; correction and distant are non-negative
    # (allow tiny floating-point noise, ~1e-17).
    assert dlpno.e_corr_iterated < 0
    assert dlpno.e_pno_correction > -1e-14
    assert dlpno.e_distant > -1e-14
    # Total correlation decomposes correctly.
    assert dlpno.e_corr == pytest.approx(
        dlpno.e_corr_iterated + dlpno.e_pno_correction + dlpno.e_distant,
        abs=1e-14,
    )
    # PNO counts are positive.
    assert dlpno.n_pairs > 0


# ---------------------------------------------------------------------------
# Stage 5c extended — DLPNO-CCSD and DLPNO-CCSD(T)
# ---------------------------------------------------------------------------


def test_toroidal_dlpno_ccsd_converges():
    """The historical periodic DLPNO-CCSD default stays all-electron."""
    from vibeqc.periodic_toroidal_mp2 import toroidal_dlpno_ccsd, toroidal_dlpno_mp2

    system = _h2_chain_cell()
    r = toroidal_dlpno_ccsd(system, "sto-3g", (1, 1, 2))
    assert r.cc_result.converged
    assert r.cc_result.n_frozen == 0
    assert r.cc_result.n_iter <= 30
    assert r.e_corr < 0
    assert r.e_corr_per_cell < 0
    # CCSD recovers more correlation than MP2.
    r_mp = toroidal_dlpno_mp2(system, "sto-3g", (1, 1, 2))
    assert r.e_corr < r_mp.e_corr  # more negative = more correlation


def test_toroidal_dlpno_ccsd_t_triples_small():
    """(T) correction is small for H2 (2-electron system)."""
    from vibeqc.periodic_toroidal_mp2 import toroidal_dlpno_ccsd_t

    system = _h2_chain_cell()
    r = toroidal_dlpno_ccsd_t(system, "sto-3g", (1, 1, 2))
    assert r.cc_result.converged
    assert r.e_t is not None
    # (T) is tiny for a 2-electron system.
    assert abs(r.e_t_per_cell) < 1e-4
    # Total energy = HF + CCSD corr + (T).
    assert r.e_total == pytest.approx(r.e_hf + r.e_corr + (r.e_t or 0.0), abs=1e-14)


# ---------------------------------------------------------------------------
# Toroidal TDL extrapolation
# ---------------------------------------------------------------------------


def test_toroidal_tdl_extrapolation():
    """Toroidal TDL extrapolation: increments converge, residuals small."""
    from vibeqc.periodic_toroidal_mp2 import toroidal_mp2_tdl

    system = _h2_chain_cell()
    r = toroidal_mp2_tdl(system, "sto-3g", (2, 3, 4, 5, 6))
    assert r.e_corr_per_cell_bulk < 0
    # Increments converge (monotonic change in last 3).
    incs = r.increments
    assert abs(incs[-1] - incs[-2]) < abs(incs[-2] - incs[-3]), (
        f"increments should converge: {incs}"
    )
    # Fit residual is small relative to bulk.
    assert r.fit_max_residual < abs(r.e_corr_per_cell_bulk) * 0.01


def test_toroidal_tdl_agrees_with_pyscf_kmp2_tdl():
    """Toroidal TDL bulk == PySCF KMP2 TDL (Task 1 pinned value)."""
    from vibeqc.periodic_toroidal_mp2 import toroidal_mp2_tdl

    from examples.regression.core.runner_pyscf import run_periodic_mp2

    system = _h2_chain_cell()
    r = toroidal_mp2_tdl(system, "sto-3g", (2, 3, 4, 5, 6))
    b_toroidal = r.e_corr_per_cell_bulk

    # Compare against the pinned PySCF KMP2 TDL from Task 1.
    # The open-megacell TDL is -0.0132746; toroidal should approach the
    # same limit.  The finite-size error in the toroidal route (BvK with
    # Ewald exxdiv) means the extrapolation has a small systematic shift.
    # We check that the KMP2 oracle at the finest mesh is close.
    bohr_to_ang = 0.52917721067
    lattice_ang = np.diag([6.0, 20.0, 20.0]) * bohr_to_ang
    atoms_frac = [
        ("H", (2.3 / 6.0, 0.5, 0.5)),
        ("H", (3.7 / 6.0, 0.5, 0.5)),
    ]
    ref = run_periodic_mp2(
        lattice_ang=lattice_ang,
        atoms_frac=atoms_frac,
        basis="sto-3g",
        kmesh=(16, 1, 1),
        dimension=1,
        low_dim_ft_type="inf_vacuum",
    )
    if ref.status == "unavailable":
        pytest.skip("PySCF not importable in the external interpreter")
    assert ref.converged
    # Toroidal TDL should agree with KMP2 at largest k-mesh within ~0.5 mHa.
    assert abs(b_toroidal - ref.e_corr) < 5e-4


# ---------------------------------------------------------------------------
# Larger-system validation — LiH chain
# ---------------------------------------------------------------------------


def _lih_chain_cell(spacing: float = 8.0, transverse: float = 20.0):
    """One LiH per cell; periodic along z; vacuum in x,y."""
    return vq.PeriodicSystem(
        3,
        np.diag([transverse, transverse, spacing]),
        [
            vq.Atom(3, [transverse / 2, transverse / 2, spacing / 2 - 1.5]),
            vq.Atom(1, [transverse / 2, transverse / 2, spacing / 2 + 1.5]),
        ],
    )


def test_toroidal_mp2_lih_structure():
    """Toroidal supercell structures for LiH chain are sensible."""
    from vibeqc.periodic_toroidal_mp2 import build_toroidal_supercell

    system = _lih_chain_cell()
    sc = build_toroidal_supercell(system, (1, 1, 2))
    atoms = list(sc.unit_cell)
    assert len(atoms) == 4  # 2 cells x 2 atoms
    # All atoms within super-lattice.
    lat = np.asarray(sc.lattice, dtype=float)
    for a in atoms:
        frac = np.linalg.solve(lat, np.asarray(a.xyz, dtype=float))
        assert np.all((frac >= -1e-12) & (frac < 1.0 + 1e-12))


# ---------------------------------------------------------------------------
# G-PBC-008: toroidal pair distances consumed by the DLPNO drivers
# ---------------------------------------------------------------------------


def test_toroidal_pair_distance_fn_minimum_image():
    """The pair-distance factory returns the exact toroidal minimum-image
    distance, matching the integer-cell sphere-decode oracle on
    orthorhombic and skew supercells and folding wrap-around separations
    onto the torus."""
    from vibeqc.periodic_toroidal_mp2 import toroidal_pair_distance_fn

    system = _h2_chain_cell(spacing_bohr=6.0)
    fn = toroidal_pair_distance_fn(system, (1, 1, 4))

    # Wrap-around fold: Euclidean 21 bohr -> torus 3 bohr (period 24).
    assert fn([0, 0, 0.5], [0, 0, 21.5]) == pytest.approx(3.0, abs=1e-12)
    # Half-period is the maximum: Euclidean 12 stays 12.
    assert fn([0, 0, 0.5], [0, 0, 12.5]) == pytest.approx(12.0, abs=1e-12)
    assert fn([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=0)

    # Consistency with the integer-cell minimum-image oracle.
    lattice = np.asarray(system.lattice, dtype=float)
    search = _prepare_toroidal_image_search(lattice @ np.diag([1.0, 1.0, 4.0]))
    for L in [(0, 0, 1), (0, 0, 2), (0, 0, 3)]:
        d_int = float(
            np.linalg.norm(_toroidal_minimum_image(L, (1, 1, 4), search))
        )
        d_cont = fn([0.0, 0.0, 0.0], lattice @ np.array(L, dtype=float))
        assert d_cont == pytest.approx(d_int, abs=1e-10)

    # Skew supercell: continuous path equals the integer oracle.
    skew = np.array(
        [[6.0, 1.5, 0.0], [0.0, 20.0, 2.0], [0.0, 0.0, 20.0]]
    )
    sk_sys = vq.PeriodicSystem(3, skew, [vq.Atom(1, [0.0, 0.0, 0.0])])
    fn_sk = toroidal_pair_distance_fn(sk_sys, (3, 1, 1))
    search_sk = _prepare_toroidal_image_search(skew @ np.diag([3.0, 1.0, 1.0]))
    for L in [(1, 0, 0), (2, 0, 0)]:
        d_int = float(
            np.linalg.norm(_toroidal_minimum_image(L, (3, 1, 1), search_sk))
        )
        d_cont = fn_sk([0.0, 0.0, 0.0], skew @ np.array(L, dtype=float))
        assert d_cont == pytest.approx(d_int, abs=1e-10)


def test_toroidal_dlpno_mp2_torus_screening_beats_euclidean():
    """The DLPNO wrapper consumes the toroidal pair geometry (G-PBC-008).

    On the (1,1,4) H2 chain the open-cluster Euclidean centroid distance
    sees the wrap-around pair at ~18 bohr (torus: 6 bohr), so the
    historical screen falsely treats a chemically relevant pair at the
    dipole-estimate level. Measured on the periodic-Ewald HF route
    (2026-08-25):

    * exact (tcut_pairs=0):      e_corr = -0.0538171385 Ha
    * torus-aware (default):     2 pairs screened, |dE| = 1.92e-5 Ha
    * forced Euclidean:          3 pairs screened, |dE| = 9.27e-5 Ha
    * extra screened wrap pair:  full contribution = -7.35e-5 Ha

    Moving toroidal HF from the molecular-limit fitted J/K branch to the
    periodic Ewald-J/real-space-K branch changed the orbitals, Fock
    denominators, and pair energies. The old aggregate ``|dE| > 1e-4``
    oracle was therefore cancellation-sensitive even though the Euclidean
    screen still drops exactly one material wrap pair. Gate that causal
    pair inventory directly, retain the 5e-5 Ha torus correctness bound,
    and require the torus result to remain closer to the exact reference.
    """
    from vibeqc.dlpno.mp2 import DLPNOMP2Options
    from vibeqc.periodic_toroidal_mp2 import toroidal_dlpno_mp2

    system = _h2_chain_cell(spacing_bohr=6.0)
    common = dict(aux_basis="def2-universal-jkfit")
    legacy_thresholds = dict(
        n_frozen=0,
        tcut_pno=1e-8,
        tcut_pno_weak=1e-7,
        tcut_mkn=1e-3,
        tcut_pairs=1e-6,
        tcut_pairs_weak=1e-4,
        # tcut_mkn=1e-3 here means Mulliken domains (#190 made DOI the
        # molecular default).
        domain_criterion="mulliken",
    )

    r_exact = toroidal_dlpno_mp2(
        system,
        "sto-3g",
        (1, 1, 4),
        dlpno_options=DLPNOMP2Options(
            **{
                **legacy_thresholds,
                "tcut_pairs": 0.0,
                "tcut_pairs_weak": 0.0,
            }
        ),
        **common,
    )
    r_torus = toroidal_dlpno_mp2(
        system,
        "sto-3g",
        (1, 1, 4),
        dlpno_options=DLPNOMP2Options(**legacy_thresholds),
        **common,
    )

    def _euclid(r_i, r_j):
        return float(np.linalg.norm(np.asarray(r_i) - np.asarray(r_j)))

    r_eucl = toroidal_dlpno_mp2(
        system,
        "sto-3g",
        (1, 1, 4),
        dlpno_options=DLPNOMP2Options(
            **legacy_thresholds, pair_distance_fn=_euclid
        ),
        **common,
    )

    n_torus = r_torus.dlpno_result.n_pairs_screened
    n_eucl = r_eucl.dlpno_result.n_pairs_screened
    assert n_eucl == n_torus + 1, (
        f"expected Euclidean screening to drop exactly one additional "
        f"wrap pair, got torus={n_torus}, Euclidean={n_eucl}"
    )

    torus_pairs = set(r_torus.dlpno_result.pair_energies)
    euclid_pairs = set(r_eucl.dlpno_result.pair_energies)
    extra_euclid_screened = torus_pairs - euclid_pairs
    assert len(extra_euclid_screened) == 1, (
        "forced-Euclidean screening must remove exactly one pair retained "
        f"by the torus geometry, got {sorted(extra_euclid_screened)}"
    )
    wrap_pair = extra_euclid_screened.pop()
    exact_wrap_energy = r_exact.dlpno_result.pair_energies[wrap_pair]
    torus_wrap_energy = r_torus.dlpno_result.pair_energies[wrap_pair]
    assert exact_wrap_energy < -5e-5, (
        f"extra Euclidean-screened pair {wrap_pair} is no longer a material "
        f"correlation contribution ({exact_wrap_energy:.3e} Ha)"
    )
    assert torus_wrap_energy == pytest.approx(exact_wrap_energy, abs=1e-6)

    d_torus = abs(r_torus.e_corr - r_exact.e_corr)
    d_eucl = abs(r_eucl.e_corr - r_exact.e_corr)
    assert d_torus < 5e-5, (
        f"torus-aware screening error {d_torus:.3e} Ha exceeds the "
        "pinned 5e-5 band"
    )
    assert d_torus < d_eucl

    cls = r_torus.pair_classification
    assert cls is not None
    assert cls.nrep == (1, 1, 4)
    assert cls.n_strong + cls.n_weak + cls.n_distant == cls.n_cells


def test_toroidal_dlpno_ccsd_coupling_uses_torus_distance():
    """The CC wrapper's coupling-radius occupied sets use the toroidal
    minimum-image distance: at coupling_radius=7 bohr on the (1,1,4)
    chain the wrap-around neighbour (torus 6 bohr, Euclidean ~18 bohr)
    is coupled under the torus metric but dropped under the Euclidean
    one, so the mean coupled-occupied set is strictly larger."""
    from dataclasses import replace

    from vibeqc.periodic_toroidal_mp2 import (
        _pre_sweep_toroidal_dlpno_cc_options,
        toroidal_dlpno_ccsd,
    )

    system = _h2_chain_cell(spacing_bohr=6.0)
    common = dict(aux_basis="def2-universal-jkfit")

    r_torus = toroidal_dlpno_ccsd(
        system,
        "sto-3g",
        (1, 1, 4),
        cc_options=replace(
            _pre_sweep_toroidal_dlpno_cc_options(),
            coupling_radius=7.0,
        ),
        **common,
    )

    def _euclid(r_i, r_j):
        return float(np.linalg.norm(np.asarray(r_i) - np.asarray(r_j)))

    r_eucl = toroidal_dlpno_ccsd(
        system,
        "sto-3g",
        (1, 1, 4),
        cc_options=replace(
            _pre_sweep_toroidal_dlpno_cc_options(),
            coupling_radius=7.0,
            pair_distance_fn=_euclid,
        ),
        **common,
    )

    avg_torus = float(r_torus.cc_result.avg_coupled_occ)
    avg_eucl = float(r_eucl.cc_result.avg_coupled_occ)
    assert avg_torus > avg_eucl, (
        f"torus coupling sets ({avg_torus:.2f}) not larger than "
        f"Euclidean ({avg_eucl:.2f}) at coupling_radius=7 -- the "
        "wrap-around neighbour was not recovered"
    )
    assert r_torus.pair_classification is not None


@pytest.mark.parametrize("custom_distance,fail", [(False, False), (True, False), (False, True)])
def test_toroidal_mp2_interleaved_sources_keep_overlap_and_options_local(
    monkeypatch, custom_distance, fail,
):
    from dataclasses import asdict
    from types import SimpleNamespace

    import vibeqc._vibeqc_core as core
    import vibeqc.dlpno.mp2 as solver
    import vibeqc.periodic_df as periodic_df
    import vibeqc.periodic_rhf_gdf as rhf_gdf
    import vibeqc.periodic_toroidal_mp2 as toroidal
    from vibeqc.dlpno.source import LpqFactors

    system = _h2_chain_cell()
    other_basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    compute_overlap = core.compute_overlap
    molecular_overlap = np.asarray(compute_overlap(other_basis))
    overlaps = [np.diag([1., 1.1, 1.2, 1.3]), np.diag([1.4, 1.5, 1.6, 1.7])]
    refs = [SimpleNamespace(
        energy=-2., converged=True, overlap=S,
        fock=np.diag([-1., -.9, .5, .6]),
        mo_coeffs=np.diag(1 / np.sqrt(S.diagonal())), exchange_q0="molecular-limit",
    ) for S in overlaps]
    queued_refs = iter(refs)
    factors = LpqFactors(np.ones((1, 4, 4)))
    factors.n_fit = 1
    factors.nbf = 4
    monkeypatch.setattr(rhf_gdf, "_bulk_gamma_k_gdf", lambda *a, **k: next(queued_refs))
    monkeypatch.setattr(periodic_df, "build_periodic_gamma_df", lambda *a, **k: factors)
    distance = (lambda a, b: float(np.linalg.norm(a-b))) if custom_distance else None
    options = solver.DLPNOMP2Options(n_frozen=1, pair_distance_fn=distance)
    before = asdict(options)
    seen = []

    class SolverFailure(Exception):
        pass

    def solve(mol, basis, reference, df, options, *, source=None):
        index = len(seen)
        seen.append(source)
        assert core.compute_overlap is compute_overlap
        np.testing.assert_array_equal(core.compute_overlap(other_basis), molecular_overlap)
        np.testing.assert_array_equal(source.overlap_matrix(), overlaps[index])
        assert source.reference is refs[index]
        assert source.reference_kind == "rhf"
        assert not source.conventions.same_source
        assert source.conventions.exchange_q0 == "molecular-limit"
        assert source.conventions.kmesh == (1, 1, 2)
        assert source.pair_distance_fn(np.zeros(3), np.array([0., 0., 11.])) == pytest.approx(1.)
        assert options.pair_distance_fn is distance
        assert options.n_frozen == 1
        if index == 0:
            toroidal.toroidal_dlpno_mp2(system, "sto-3g", (1, 1, 2), dlpno_options=options)
            np.testing.assert_array_equal(source.overlap_matrix(), overlaps[0])
            if fail:
                raise SolverFailure
        return SimpleNamespace(e_corr=-.02, source_kind=source.kind,
                               source_identity=source.identity)

    monkeypatch.setattr(solver, "run_dlpno_mp2", solve)
    if fail:
        with pytest.raises(SolverFailure):
            toroidal.toroidal_dlpno_mp2(system, "sto-3g", (1, 1, 2), dlpno_options=options)
    else:
        result = toroidal.toroidal_dlpno_mp2(system, "sto-3g", (1, 1, 2), dlpno_options=options)
        assert result.source_kind == result.dlpno_result.source_kind == "toroidal-compcell"
        assert result.source_identity == seen[0].identity
        assert not result.same_source
        assert result.exchange_q0 == "molecular-limit"
        assert result.e_corr_per_cell == pytest.approx(-.01)
        assert result.e_total == pytest.approx(-2.02)
    assert len(seen) == 2
    assert seen[0].identity != seen[1].identity
    assert asdict(options) == before
    assert core.compute_overlap is compute_overlap
    for ref, S in zip(refs, overlaps):
        np.testing.assert_array_equal(ref.overlap, S)


@pytest.mark.parametrize("with_triples,custom_distance,fail", [
    (False, False, False), (False, True, False),
    (True, False, False), (True, True, False), (True, True, True),
])
def test_toroidal_cc_keeps_overlap_and_options_local(
    monkeypatch, with_triples, custom_distance, fail,
):
    """An interleaved molecular calculation must keep its own AO metric."""
    from types import SimpleNamespace

    import vibeqc._vibeqc_core as core
    import vibeqc.dlpno.ccsd_local_solver as solver
    import vibeqc.periodic_df as periodic_df
    import vibeqc.periodic_rhf_gdf as rhf_gdf
    import vibeqc.periodic_toroidal_mp2 as toroidal
    from vibeqc.dlpno.source import LpqFactors

    system = _h2_chain_cell()
    other_basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    compute_overlap = core.compute_overlap
    molecular_overlap = np.asarray(compute_overlap(other_basis))
    S = np.diag([1.0, 1.1, 1.2, 1.3])
    hf = SimpleNamespace(
        energy=-2.0, converged=True, overlap=S, fock=np.diag([-1., -.9, .5, .6]),
        mo_coeffs=np.diag(1 / np.sqrt(S.diagonal())), exchange_q0="molecular-limit",
    )
    factors = LpqFactors(np.ones((1, 4, 4)))
    factors.n_fit = 1
    factors.nbf = 4
    monkeypatch.setattr(rhf_gdf, "_bulk_gamma_k_gdf", lambda *a, **k: hf)
    monkeypatch.setattr(periodic_df, "build_periodic_gamma_df", lambda *a, **k: factors)
    distance = (lambda a, b: float(np.linalg.norm(a-b))) if custom_distance else None
    options = solver.LocalCCSDOptions(
        compute_triples=not with_triples, pair_distance_fn=distance,
    )

    class SolverFailure(Exception):
        pass

    def solve(mol, basis, reference, df, options, *, source=None):
        assert options.compute_triples == with_triples
        assert core.compute_overlap is compute_overlap
        np.testing.assert_array_equal(core.compute_overlap(other_basis), molecular_overlap)
        np.testing.assert_array_equal(source.overlap_matrix(), S)
        assert source.reference is hf
        assert not source.conventions.same_source
        assert source.conventions.exchange_q0 == "molecular-limit"
        assert source.conventions.kmesh == (1, 1, 2)
        assert source.pair_distance_fn(np.zeros(3), np.array([0., 0., 11.])) == pytest.approx(1.)
        assert options.pair_distance_fn is distance
        if fail:
            raise SolverFailure
        return SimpleNamespace(
            e_corr=-.02, e_t=-.001 if options.compute_triples else 0.,
            source_kind=source.kind, source_identity=source.identity,
        )

    monkeypatch.setattr(solver, "run_local_dlpno_ccsd", solve)
    driver = toroidal.toroidal_dlpno_ccsd_t if with_triples else toroidal.toroidal_dlpno_ccsd
    if fail:
        with pytest.raises(SolverFailure):
            driver(system, "sto-3g", (1, 1, 2), cc_options=options)
    else:
        result = driver(system, "sto-3g", (1, 1, 2), cc_options=options)
        assert result.source_kind == result.cc_result.source_kind == "toroidal-compcell"
        assert result.source_identity == result.cc_result.source_identity
        assert len(result.source_identity) == 64
        assert not result.same_source
        assert result.exchange_q0 == "molecular-limit"
        assert result.e_t == (-.001 if with_triples else None)
        assert result.e_total == pytest.approx(-2.02 - (.001 if with_triples else 0))
    assert options.compute_triples == (not with_triples)
    assert options.pair_distance_fn is distance
    assert core.compute_overlap is compute_overlap
