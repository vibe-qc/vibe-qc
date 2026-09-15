"""Phase 12f tests: periodic Becke partition for DFT grid integration.

Contracts exercised:

1. **Molecular limit equality** — at ``image_radius_bohr = 0`` the
   periodic build returns exactly the molecular ``build_grid`` (same
   points, same weights), to bit-precision.

2. **Localized molecular limit** — a converged physical point neighborhood
   preserves molecular integrals of a localized Gaussian density.

3. **Tight-cell volume** — on a cell smaller than the partition reach
   the periodic-Becke total weight integrates the unit-cell volume
   ``Σ_g w_g ≈ V_cell``, while the molecular partition over-counts
   substantially. This is the central physical contract: the periodic
   partition restores ∫_cell normalisation.

4. **Extended atom list shape** — :func:`extended_partition_atoms`
   places the home-cell atoms first in the original order, with image
   atoms following.

5. **Bad input rejection** — negative radius and degenerate lattices
   raise ``ValueError``.

6. **build_xc_periodic integration** — the resulting grid is
   structurally compatible with the existing periodic XC builder.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _h2_loose(box: float = 30.0):
    c = box / 2
    return vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )


def _h2_tight(a: float = 5.0):
    return vq.PeriodicSystem(
        3, np.eye(3) * a,
        [vq.Atom(1, [0, 0, 0]),
         vq.Atom(1, [a/2, a/2, a/2])],
    )


def _small_grid_options():
    """Reduced-resolution grid for fast tests."""
    opts = vq.GridOptions()
    opts.n_radial = 25
    opts.n_theta = 7
    opts.n_phi = 12
    return opts


# ---------------------------------------------------------------------------
# 1. Molecular-limit equality at image_radius = 0
# ---------------------------------------------------------------------------

def test_zero_radius_reduces_to_molecular_grid():
    sysp = _h2_loose()
    grid_opts = _small_grid_options()
    g_mol = vq.build_grid(sysp.unit_cell_molecule(), grid_opts)
    g_per = vq.build_periodic_becke_grid(
        sysp, grid_options=grid_opts, image_radius_bohr=0.0,
    )
    assert g_mol.points.shape == g_per.points.shape
    assert np.allclose(g_mol.points, g_per.points, atol=0.0)
    assert np.allclose(g_mol.weights, g_per.weights, atol=0.0)
    assert list(g_mol.atom_of_point) == list(g_per.atom_of_point)


# ---------------------------------------------------------------------------
# 2. Localized molecular limit with a converged physical radius
# ---------------------------------------------------------------------------

def test_loose_box_periodic_matches_localized_molecular_integral():
    """Raw grids agree; the physical radial support has a negligible tail."""
    sysp = _h2_loose(box=30.0)
    grid_opts = _small_grid_options()
    g_mol = vq.build_grid(sysp.unit_cell_molecule(), grid_opts)
    g_per = vq.build_periodic_becke_grid(
        sysp, grid_options=grid_opts, image_radius_bohr=8.0,
    )
    np.testing.assert_array_equal(g_mol.points, g_per.points)
    np.testing.assert_array_equal(g_mol.atomic_weights, g_per.atomic_weights)
    points = np.asarray(g_mol.points)
    origins = np.array([atom.xyz for atom in sysp.unit_cell])
    density = (2/np.pi)**1.5*np.exp(
        -2*np.sum((points[:, None, :]-origins[None, :, :])**2, axis=-1),
    ).sum(axis=1)
    assert float(np.asarray(g_per.weights)@density) == pytest.approx(
        float(np.asarray(g_mol.weights)@density), rel=0, abs=2e-12,
    )


def test_periodic_partition_preserves_diffuse_density_in_large_vacuum():
    """An image cutoff must not silently become an atomic radial cutoff."""
    system = vq.PeriodicSystem(3, 1000.*np.eye(3), [vq.Atom(2, [0., 0., 0.])])
    opts = vq.GridOptions()
    opts.n_radial = 300
    opts.n_theta = 5
    opts.n_phi = 8
    molecular = vq.build_grid(system.unit_cell_molecule(), opts)
    periodic = vq.build_periodic_becke_grid(system, grid_options=opts, image_radius_bohr=10.)
    points = np.asarray(molecular.points)
    alpha = .005
    density = 2.*(2*alpha/np.pi)**1.5*np.exp(-2*alpha*np.sum(points*points, axis=1))
    expected = float(np.asarray(molecular.weights)@density)
    actual = float(np.asarray(periodic.weights)@density)
    assert expected > 1.9
    assert actual == pytest.approx(expected, rel=0, abs=2e-12)


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("profile", ["becke", "stratmann", "pyscf-level3"])
def test_point_partition_preserves_image_labels_translation_and_raw_grid(dimension, profile):
    lattice = np.array([[6., .3, .1], [.4, 8., .2], [.1, .2, 9.]])
    lattice[:, dimension:] = 0.
    positions = np.array([[.1, .2, .3], [1.4, .3, .1]])
    opts = _small_grid_options()
    if profile == "pyscf-level3":
        opts.atomic_grid_profile = profile
    else:
        opts.partition = profile

    def build(atoms):
        system = vq.PeriodicSystem(dimension, lattice, [
            vq.Atom(z, p) for z, p in zip([1, 3], atoms)
        ])
        return vq.build_periodic_becke_grid(system, grid_options=opts, image_radius_bohr=10.)

    original = build(positions)
    owners = np.asarray(original.atom_of_point)
    for image_shift, rigid_shift in [(3, np.zeros(3)), (0, np.array([.31, -.27, .19]))]:
        shifted = positions+rigid_shift
        shifted[1] += image_shift*lattice[:, 0]
        actual = build(shifted)
        displacements = shifted-positions
        np.testing.assert_allclose(
            actual.points, np.asarray(original.points)+displacements[owners],
            rtol=0, atol=3e-13,
        )
        # The partition factor is bounded by one; raw vacuum radial weights
        # can be large. Compare the factor directly and pin raw weights exactly.
        np.testing.assert_allclose(
            np.asarray(actual.weights)/np.asarray(actual.atomic_weights),
            np.asarray(original.weights)/np.asarray(original.atomic_weights),
            rtol=0, atol=3e-14,
        )
        np.testing.assert_array_equal(actual.atomic_weights, original.atomic_weights)
        np.testing.assert_array_equal(actual.atom_of_point, original.atom_of_point)
        np.testing.assert_array_equal(actual.atomic_numbers, original.atomic_numbers)
        np.testing.assert_array_equal(actual.atom_coords, shifted)


def test_point_partition_is_invariant_to_skew_lattice_reparametrization():
    lattice = np.array([[6., .3, .1], [.4, 8., .2], [.1, .2, 9.]])
    unimodular = np.array([[1, 7, 0], [0, 1, 0], [0, 0, 1]])
    atoms = [vq.Atom(1, [.1, .2, .3]), vq.Atom(1, [1.4, .3, .1])]
    opts = _small_grid_options()
    opts.n_radial = 12
    opts.n_theta = 5
    opts.n_phi = 8
    original = vq.build_periodic_becke_grid(
        vq.PeriodicSystem(3, lattice, atoms), grid_options=opts, image_radius_bohr=1.,
    )
    skew = vq.build_periodic_becke_grid(
        vq.PeriodicSystem(3, lattice@unimodular, atoms), grid_options=opts, image_radius_bohr=1.,
    )
    np.testing.assert_array_equal(skew.points, original.points)
    np.testing.assert_allclose(skew.weights, original.weights, rtol=0, atol=3e-12)


def test_point_partition_matches_independent_periodic_image_normalization():
    from itertools import product

    lattice = np.array([[6., .3, .1], [.4, 8., .2], [.1, .2, 9.]])
    origins = np.array([[.1, .2, .3], [1.4, .3, .1]])
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, p) for p in origins])
    opts = _small_grid_options()
    radius = 10.
    grid = vq.build_periodic_becke_grid(system, grid_options=opts, image_radius_bohr=radius)
    # Oversized, independent image enumeration followed by the physical
    # point predicate; normalized products follow Becke 1988 Eqs. 13/22.
    indices = np.array(list(product(range(-6, 7), repeat=3)))
    centers = (origins[None, :, :]+(indices@lattice.T)[:, None, :]).reshape(-1, 3)
    home = 2*np.flatnonzero(np.all(indices == 0, axis=1))[0]
    selected = np.random.default_rng(772).choice(len(grid.weights), size=64, replace=False)
    nonzero = 0
    for p in selected:
        point = np.asarray(grid.points)[p]
        owner = grid.atom_of_point[p]
        image_distances = np.linalg.norm(centers-point, axis=1)
        point_radius = max(radius, 2*float(image_distances.min()))
        if np.linalg.norm(point-origins[owner]) > point_radius:
            assert grid.weights[p] == 0.
            continue
        keep = np.flatnonzero(image_distances <= point_radius)
        distances = np.linalg.norm(centers[keep]-point, axis=1)
        weights = np.ones(len(keep))
        for i, j in product(range(len(keep)), repeat=2):
            if i == j:
                continue
            mu = (distances[i]-distances[j])/np.linalg.norm(centers[keep[i]]-centers[keep[j]])
            for _ in range(3):
                mu = .5*mu*(3-mu*mu)
            weights[i] *= .5*(1-mu)
        expected = weights[np.flatnonzero(keep == home+owner)[0]]/weights.sum()
        assert grid.weights[p]/grid.atomic_weights[p] == pytest.approx(expected, rel=0, abs=2e-13)
        nonzero += expected > 1e-5
    assert nonzero >= 10


# ---------------------------------------------------------------------------
# 3. Tight-cell volume: periodic Becke recovers ∫_cell 1 dr = V_cell
# ---------------------------------------------------------------------------

def test_tight_cell_periodic_grid_integrates_cell_volume():
    """On a tight 5-bohr cubic cell, the periodic-Becke total grid
    weight equals the unit-cell volume V_cell = 125 bohr³ to within
    grid-resolution error. The molecular partition over the same cell
    grossly over-counts because there's no image-atom denominator to
    fence partition weight at the boundary."""
    sysp = _h2_tight(a=5.0)
    grid_opts = _small_grid_options()
    V_cell = 5.0 ** 3   # 125 bohr³

    g_mol = vq.build_grid(sysp.unit_cell_molecule(), grid_opts)
    g_per = vq.build_periodic_becke_grid(
        sysp, grid_options=grid_opts, image_radius_bohr=10.0,
    )
    # Periodic: integrates the cell volume to within ~ a few %.
    rel_per = abs(g_per.weights.sum() - V_cell) / V_cell
    assert rel_per < 0.05, (
        f"periodic grid weight sum {g_per.weights.sum():.3f} vs "
        f"V_cell {V_cell}; rel_err {rel_per:.3e}"
    )
    # Molecular: over-counts by orders of magnitude.
    over_count = g_mol.weights.sum() / V_cell
    assert over_count > 10.0, (
        f"molecular Becke total weight only {g_mol.weights.sum():.3f} "
        f"on a 5-bohr cell — expected >> V_cell ({V_cell}) due to "
        "missing image-atom partition denominator"
    )


# ---------------------------------------------------------------------------
# 4. Extended atom list ordering
# ---------------------------------------------------------------------------

def test_extended_atom_list_lists_home_first():
    sysp = _h2_tight(a=5.0)
    home_positions = np.array([a.xyz for a in sysp.unit_cell])
    ext = vq.extended_partition_atoms(sysp, image_radius_bohr=10.0)
    assert len(ext) >= len(home_positions)
    for idx, home_pos in enumerate(home_positions):
        assert np.allclose(ext[idx], home_pos), (
            f"extended_partition_atoms[{idx}] = {ext[idx]} does not "
            f"match home atom {idx} = {home_pos}"
        )
    # All image atoms have positive distance from the home cell.
    for img_pos in ext[len(home_positions):]:
        d_min = min(
            float(np.linalg.norm(img_pos - h)) for h in home_positions
        )
        assert d_min > 0.0


def test_extended_atom_list_grows_with_radius():
    sysp = _h2_tight(a=5.0)
    n_home = len(sysp.unit_cell)
    n_3 = len(vq.extended_partition_atoms(sysp, image_radius_bohr=3.0))
    n_10 = len(vq.extended_partition_atoms(sysp, image_radius_bohr=10.0))
    assert n_3 >= n_home
    assert n_10 >= n_3
    # 10 bohr clearly captures more images than 3 bohr on this cell.
    assert n_10 > n_3


def test_extended_atom_list_uses_lattice_columns_for_hexagonal_cells():
    """Image atoms in skew cells must be generated from lattice columns.

    Cubic cells cannot distinguish row-vector and column-vector bugs; a
    hexagonal cell can. This is the periodic-Becke counterpart of the
    Born-von Karman convention used everywhere else in vibe-qc.
    """
    a = 5.0
    c = 18.0
    lattice = np.column_stack([
        [a, 0.0, 0.0],
        [0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
        [0.0, 0.0, c],
    ])
    sysp = vq.PeriodicSystem(3, lattice, [vq.Atom(1, [0.0, 0.0, 0.0])])

    ext = vq.extended_partition_atoms(sysp, image_radius_bohr=5.05)
    images = np.asarray(ext[1:], dtype=float)

    for shift in (
        lattice[:, 0],
        -lattice[:, 0],
        lattice[:, 1],
        -lattice[:, 1],
        lattice[:, 1] - lattice[:, 0],
        lattice[:, 0] - lattice[:, 1],
    ):
        assert np.min(np.linalg.norm(images - shift, axis=1)) < 1.0e-12


@pytest.mark.parametrize(
    ("dim", "lattice", "image_radius", "expected_images"),
    [
        (1, np.diag([2.0, 0.25, 0.5]), 2.01, 2),
        (2, np.diag([2.0, 2.0, 0.25]), 2.01, 4),
    ],
)
def test_extended_atom_list_never_repeats_nonperiodic_axes(
    dim, lattice, image_radius, expected_images
):
    """Bookkeeping lattice columns must not create physical image atoms."""
    system = vq.PeriodicSystem(
        dim, lattice, [vq.Atom(1, [0.0, 0.0, 0.0])]
    )

    images = np.asarray(
        vq.extended_partition_atoms(system, image_radius)[1:], dtype=float
    )

    assert len(images) == expected_images
    inactive = images[:, dim:]
    np.testing.assert_allclose(inactive, 0.0, rtol=0.0, atol=0.0)


def test_extended_atom_list_skew_bound_includes_cancelling_indices():
    """The coefficient box stays complete for nearly cancelling vectors."""
    lattice = np.column_stack(
        [
            [10.0, 0.0, 0.0],
            [9.9, 0.1, 0.0],
            [0.0, 0.0, 30.0],
        ]
    )
    system = vq.PeriodicSystem(
        2, lattice, [vq.Atom(1, [0.0, 0.0, 0.0])]
    )
    images = np.asarray(
        vq.extended_partition_atoms(system, image_radius_bohr=0.75)[1:],
        dtype=float,
    )

    # n=(5,-5) is outside the former [-2,2]^2 box, but its skew-vector
    # cancellation leaves a physical shift of only sqrt(1/2) bohr.
    cancelling_shift = 5.0 * lattice[:, 0] - 5.0 * lattice[:, 1]
    assert np.linalg.norm(cancelling_shift) < 0.75
    assert np.min(np.linalg.norm(images - cancelling_shift, axis=1)) < 1.0e-12


# ---------------------------------------------------------------------------
# 5. Bad input rejection
# ---------------------------------------------------------------------------

def test_negative_radius_raises():
    sysp = _h2_loose()
    with pytest.raises(ValueError, match="image_radius_bohr"):
        vq.build_periodic_becke_grid(sysp, image_radius_bohr=-1.0)


@pytest.mark.parametrize("radius", [float("nan"), float("inf")])
def test_nonfinite_periodic_partition_radius_raises(radius):
    with pytest.raises(ValueError, match="finite"):
        vq.build_periodic_becke_grid(_h2_loose(), image_radius_bohr=radius)


def test_periodic_partition_rejects_coincident_atom_images():
    lattice = np.diag([6., 8., 9.])
    system = vq.PeriodicSystem(3, lattice, [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, lattice[:, 0]),
    ])
    with pytest.raises(ValueError, match="Coincident periodic atoms"):
        vq.build_periodic_becke_grid(system, grid_options=_small_grid_options())


def test_partition_atoms_must_start_with_home_atoms():
    """If the user calls the C++ build_grid_periodic directly with a
    partition list whose first len(grid_mol.atoms()) entries don't
    match the home atoms, it raises (sanity guard against caller
    mistakes)."""
    import vibeqc._vibeqc_core as core
    sysp = _h2_loose()
    home_mol = sysp.unit_cell_molecule()
    bogus_partition = [
        np.array([99.0, 99.0, 99.0]),   # nonsense first entry
        np.array([0.0, 0.0, 0.0]),
    ]
    with pytest.raises(ValueError, match="first N_grid"):
        core.build_grid_periodic(home_mol, bogus_partition)


# ---------------------------------------------------------------------------
# 6. Integration with build_xc_periodic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6b. Default-flag regression
# ---------------------------------------------------------------------------

def test_periodic_ks_options_default_uses_periodic_becke():
    """``PeriodicKSOptions().use_periodic_becke`` is ``True`` by default
    (since v0.9.x). The previous default of ``False`` silently corrupted
    E_xc on every tight crystal — see docs/index.md § *Periodic V_xc on
    tight cells*. Localized loose-cell integrals converge to the molecular result
    as the partition radius covers the density (pinned by
    :func:`test_run_rks_periodic_periodic_becke_loose_box_equals_molecular`).
    """
    opts = vq.PeriodicKSOptions()
    assert opts.use_periodic_becke is True
    assert opts.becke_image_radius_bohr == pytest.approx(10.0)


def test_lattice_sum_options_expose_becke_image_radius():
    """``LatticeSumOptions.becke_image_radius_bohr`` is bound to Python.

    The C++ field has always existed (``lattice_sum.hpp``) and
    ``build_xc_periodic`` screens the cross-cell bra-image density by it,
    but the pybind11 ``def_readwrite`` was originally added only on
    ``PeriodicKSOptions``, so Python callers that set it on a bare
    ``LatticeSumOptions`` (the basis-optimization XC parameter gradient,
    ``tests/basisset_dev/test_periodic_xc_param_gradient.py``) crashed
    with AttributeError. Default 0.0 means unset (falls back to the
    built-in partition reach)."""
    from vibeqc import _vibeqc_core as core

    opts = core.LatticeSumOptions()
    assert opts.becke_image_radius_bohr == pytest.approx(0.0)
    opts.becke_image_radius_bohr = 7.5
    assert opts.becke_image_radius_bohr == pytest.approx(7.5)


def test_run_rks_periodic_default_flag_picks_up_periodic_becke():
    """A bare ``PeriodicKSOptions`` on a tight LiH-rocksalt primitive
    converges to the same energy as an explicit
    ``use_periodic_becke=True``, and differs from
    ``use_periodic_becke=False`` by O(10) mHa — the V_xc bug fingerprint
    from docs/index.md. If this test starts passing with the flag off it
    means the bug snuck back in (or the default flipped back to False)."""
    # LiH-rocksalt primitive cell (Fm-3m): FCC lattice vectors of
    # length a/√2 ≈ 5.46 bohr, Li at the origin and H at the body
    # centre of the conventional cubic cell. This is the canonical
    # tight crystal where the molecular Becke partition over-counts
    # by ~50 mHa/atom on the SCF energy.
    a_bohr = 4.084 / 0.529177210903
    lat = 0.5 * a_bohr * np.array(
        [[0.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]],
        dtype=float,
    ).T  # lattice vectors in columns
    sysp = vq.PeriodicSystem(
        3, lat,
        [vq.Atom(3, [0.0, 0.0, 0.0]),
         vq.Atom(1, [a_bohr * 0.5, a_bohr * 0.5, a_bohr * 0.5])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    def _opts(flag):
        o = vq.PeriodicKSOptions()
        o.functional = "LDA"
        o.max_iter = 80
        o.damping = 0.5
        o.conv_tol_energy = 1e-9
        o.conv_tol_grad = 1e-7
        o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
        o.lattice_opts.cutoff_bohr = 12.0
        o.lattice_opts.nuclear_cutoff_bohr = 15.0
        o.grid.n_radial = 50
        o.grid.n_theta = 11
        o.grid.n_phi = 22
        if flag is not None:
            o.use_periodic_becke = flag
        return o

    # Tight LiH: opt past the fail-closed dense-ionic guard to compare the
    # Becke-grid variants on a dense cell (this checks the *relative* energy
    # fingerprint, not absolute correctness).
    e_default = vq.run_rks_periodic_gamma_ewald3d(
        sysp, basis, _opts(None), allow_dense_ionic=True
    ).energy
    e_on = vq.run_rks_periodic_gamma_ewald3d(
        sysp, basis, _opts(True), allow_dense_ionic=True
    ).energy
    e_off = vq.run_rks_periodic_gamma_ewald3d(
        sysp, basis, _opts(False), allow_dense_ionic=True
    ).energy

    # Default must match the explicit-on path.
    assert e_default == pytest.approx(e_on, abs=1e-9), (
        f"default flag gives {e_default:.9f}; explicit True "
        f"gives {e_on:.9f}; the default is no longer True"
    )
    # Default must NOT match the legacy False path on a tight cell.
    # On LiH/STO-3G at the experimental lattice constant the bug is
    # roughly -100 mHa on the total energy (50 mHa/atom).  Pin > 10 mHa
    # so we trip if anyone narrows the gap without proper validation.
    assert abs(e_off - e_on) > 0.010, (
        f"Δ between use_periodic_becke True/False is "
        f"{e_off - e_on:.6f} Ha on tight LiH — expected ≳ 10 mHa. "
        "If the gap collapsed, either build_xc_periodic was actually "
        "fixed (update this test) or the partition stopped seeing "
        "image atoms (regression)."
    )


def test_run_rks_periodic_periodic_becke_loose_box_equals_molecular():
    """A converged loose-cell partition preserves the molecular SCF energy."""
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * 30.0,
        [vq.Atom(1, [15.0, 15.0, 14.3]),
         vq.Atom(1, [15.0, 15.0, 15.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [1, 1, 1])

    def _opts(periodic_becke: bool):
        o = vq.PeriodicKSOptions()
        o.functional = "LDA"
        o.lattice_opts.cutoff_bohr = 12.0
        o.lattice_opts.nuclear_cutoff_bohr = 15.0
        o.damping = 0.3
        o.max_iter = 40
        o.grid.n_radial = 25
        o.grid.n_theta = 7
        o.grid.n_phi = 12
        o.use_periodic_becke = periodic_becke
        o.becke_image_radius_bohr = 8.0
        return o

    r_mol = vq.run_rks_periodic(sysp, basis, km, _opts(False))
    r_per = vq.run_rks_periodic(sysp, basis, km, _opts(True))
    assert r_mol.converged and r_per.converged
    assert r_mol.energy == pytest.approx(r_per.energy, abs=1e-10)


def test_periodic_grid_feeds_into_build_xc_periodic():
    """The periodic-Becke grid is structurally compatible with the
    periodic XC builder (same Grid type, same fields). Invoke it on a
    trivial density and verify it produces finite, sensible output."""
    sysp = _h2_loose(box=20.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    grid_opts = _small_grid_options()
    grid = vq.build_periodic_becke_grid(
        sysp, grid_options=grid_opts, image_radius_bohr=8.0,
    )
    # Trivial density: identity-shaped real-space density on g=0 only.
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    opts.nuclear_cutoff_bohr = 15.0
    P_real = vq.compute_overlap_lattice(basis, sysp, opts)
    # Replace the g=0 block with a simple SCF-like density (ones on diag).
    nbf = basis.nbasis
    P_real.blocks[0] = 0.5 * np.eye(nbf)
    func = vq.Functional("LDA")
    contrib = vq.build_xc_periodic(basis, sysp, grid, func, P_real, opts)
    assert np.isfinite(contrib.e_xc)
    # V_xc should have the same cell list as the input density.
    assert len(contrib.V_xc.cells) == len(P_real.cells)
