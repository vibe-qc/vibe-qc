"""Slab + adsorbate builder tests.

Pure-geometry checks: lattice symmetry, atom counts, layer indexing,
adsorbate placement at named sites. No SCF is run here.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom
from vibeqc.build import (
    BULK_LATTICE_CONSTANTS,
    SlabInfo,
    _orient_molecule,
    _resolve_anchor,
    _wrap_in_cell,
    molecule,
    place_adsorbate,
    slab,
    slab_2d,
)

_BOHR_TO_ANG = 0.529177210903


def _lattice_ang(system):
    return np.asarray(system.lattice, dtype=float) * _BOHR_TO_ANG


def _positions_ang(system):
    return np.array([list(a.xyz) for a in system.unit_cell], dtype=float) * _BOHR_TO_ANG


def test_fcc111_5layer_ni_default_lattice():
    # The c = 4·d + vacuum geometry is a dim=3-with-vacuum property, so this
    # exercises the explicit periodic_z=True escape hatch.
    sys, info = slab("Ni", facet=(1, 1, 1), n_layers=5, vacuum=12.0, periodic_z=True)
    assert sys.dim == 3
    assert info.structure == "fcc"
    assert info.facet == (1, 1, 1)
    assert info.n_layers == 5
    # 1 atom per layer in the primitive surface cell × 5 layers.
    assert len(sys.unit_cell) == 5
    # Ni default a = 3.524 Å. Interlayer spacing d = a/√3 ≈ 2.035 Å.
    a = BULK_LATTICE_CONSTANTS["Ni"][1]
    d = a / np.sqrt(3)
    pos = _positions_ang(sys)
    z = np.sort(pos[:, 2])
    spacings = np.diff(z)
    assert np.allclose(spacings, d, atol=1e-6)
    # Vacuum: c_len - (n_layers - 1) * d should match the requested 12 Å.
    c = _lattice_ang(sys)[2, 2]
    assert np.isclose(c - 4 * d, 12.0, atol=1e-6)


def test_bcc100_fe_supercell_layers_and_z_centring():
    # z-centring in a vacuum box is dim=3-with-vacuum behaviour (periodic_z=True).
    sys, info = slab(
        "Fe", facet=(1, 0, 0), n_layers=5, vacuum=12.0, supercell=(2, 2),
        periodic_z=True,
    )
    assert sys.dim == 3
    # 1 atom/layer × 5 layers × 4 (2×2 supercell) = 20 atoms.
    assert len(sys.unit_cell) == 20
    assert info.supercell == (2, 2)
    # All layer indices represented exactly 4 times (one per supercell tile).
    counts = np.bincount(info.layer_index, minlength=5)
    assert counts.tolist() == [4, 4, 4, 4, 4]
    # The slab is centred along z: top vacuum == bottom vacuum.
    pos = _positions_ang(sys)
    c = _lattice_ang(sys)[2, 2]
    top_vac = c - pos[:, 2].max()
    bot_vac = pos[:, 2].min()
    assert np.isclose(top_vac, bot_vac, atol=1e-6)


def test_slab_default_is_dim2_vacuum_free():
    # New default: slab() builds a genuine dim=2 layer (no vacuum-as-periodicity,
    # atoms keep their real z starting at 0). Layer geometry is unchanged.
    sys, info = slab("Ni", facet=(1, 1, 1), n_layers=5, vacuum=12.0)
    assert sys.dim == 2
    assert len(sys.unit_cell) == 5
    a = BULK_LATTICE_CONSTANTS["Ni"][1]
    d = a / np.sqrt(3)
    pos = _positions_ang(sys)
    z = np.sort(pos[:, 2])
    assert np.allclose(np.diff(z), d, atol=1e-6)
    # Atoms are NOT centred in a vacuum box: the bottom layer sits at z=0.
    assert np.isclose(z[0], 0.0, atol=1e-6)
    # The synthesized a3 is a normal-direction bookkeeping vector, decoupled
    # from `vacuum`; its length is >= 30 bohr and does not equal slab+vacuum.
    a3 = np.linalg.norm(np.asarray(sys.lattice, dtype=float)[:, 2])  # bohr
    assert a3 >= 30.0 - 1e-9


def test_slab_2d_from_lattice_vectors_and_a3_invariance():
    box = 18.0
    atoms = [Atom(1, [9.0, 9.0, -0.7]), Atom(1, [9.0, 9.0, 0.7])]
    s = slab_2d([box, 0.0, 0.0], [0.0, box, 0.0], atoms)
    assert s.dim == 2
    assert len(s.unit_cell) == 2
    lat = np.asarray(s.lattice, dtype=float)
    assert np.isclose(np.linalg.norm(lat[:, 0]), box)
    assert np.isclose(np.linalg.norm(lat[:, 1]), box)
    # a3 is along the normal (+z here) and orthogonal to the plane.
    assert np.isclose(lat[0, 2], 0.0, atol=1e-9)
    assert np.isclose(lat[1, 2], 0.0, atol=1e-9)
    # a3 length policy: max(30, z_extent + 2*15) = max(30, 1.4/BOHR + 30).
    assert np.linalg.norm(lat[:, 2]) >= 30.0 - 1e-9
    # Collinear in-plane vectors are rejected.
    with pytest.raises(ValueError):
        slab_2d([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], atoms)


def test_lattice_is_columns_in_bohr():
    """Sanity: PeriodicSystem.lattice columns are a, b, c (bohr)."""
    sys, info = slab("Cu", facet=(1, 0, 0), n_layers=3, vacuum=10.0)
    lat = np.asarray(sys.lattice, dtype=float) * _BOHR_TO_ANG
    a_cu = BULK_LATTICE_CONSTANTS["Cu"][1]
    # fcc(100) primitive surface cell side = a/√2 ≈ 2.557 Å for Cu.
    expected = a_cu / np.sqrt(2)
    assert np.isclose(np.linalg.norm(lat[:, 0]), expected, atol=1e-6)
    assert np.isclose(np.linalg.norm(lat[:, 1]), expected, atol=1e-6)
    # The two in-plane vectors are orthogonal for fcc(100).
    assert np.isclose(lat[:2, 0] @ lat[:2, 1], 0.0, atol=1e-8)


def test_hcp_basal_two_atoms_in_three_layers():
    sys, info = slab("Mg", facet=(0, 0, 0, 1), n_layers=3, vacuum=12.0)
    # 1 atom/layer × 3 layers = 3.
    assert info.facet == (0, 0, 1)
    assert len(sys.unit_cell) == 3
    # ABAB stacking — layer 2's in-plane offset matches layer 0.
    pos = _positions_ang(sys)
    order = np.argsort(pos[:, 2])
    pos = pos[order]
    assert np.allclose(pos[0, :2], pos[2, :2], atol=1e-6)
    assert not np.allclose(pos[0, :2], pos[1, :2], atol=1e-3)


def test_bottom_layer_indices_for_freeze():
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=5, vacuum=12.0, supercell=(2, 2))
    # 4 atoms per layer × 5 layers = 20.
    assert len(sys.unit_cell) == 20
    freeze = info.bottom_layer_indices(3)
    assert len(freeze) == 12
    # The frozen atoms are exactly those at the bottom three layers.
    layer_set = set(info.layer_index[i] for i in freeze)
    assert layer_set == {0, 1, 2}


def test_unknown_element_requires_explicit_a():
    with pytest.raises(ValueError, match="no default structure"):
        slab("Hf", facet=(1, 0, 0), n_layers=3, vacuum=10.0)
    # Explicit a + structure works.
    sys, info = slab(
        "Hf",
        structure="hcp",
        facet=(0, 0, 1),
        n_layers=3,
        vacuum=10.0,
        a=3.195,
        c=5.051,
    )
    assert info.element == "Hf"
    assert info.a == 3.195


def test_molecule_diatomic_bond_length_override():
    n2 = molecule("N2", bond_length=1.10)
    syms = [s for s, _ in n2]
    assert syms == ["N", "N"]
    xs = np.array([r for _, r in n2])
    assert np.isclose(np.linalg.norm(xs[1] - xs[0]), 1.10, atol=1e-10)


# ---------------------------------------------------------------------------
# Existing place_adsorbate tests (backward compatibility)
# ---------------------------------------------------------------------------


def test_place_adsorbate_top_height_and_atom_count():
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    n_slab = len(sys.unit_cell)
    sys2 = place_adsorbate(
        sys, "CO", info=info, site="top", height=2.0, orientation="end-on"
    )
    assert len(sys2.unit_cell) == n_slab + 2
    # Find the C (Z=6) and O (Z=8) — they're the last two atoms.
    ads = list(sys2.unit_cell)[n_slab:]
    zs = [a.Z for a in ads]
    assert zs == [6, 8]
    # C sits exactly 2.0 Å above the top slab layer's z.
    z_top_ang = max(a.xyz[2] for a in sys.unit_cell) * _BOHR_TO_ANG
    z_c_ang = ads[0].xyz[2] * _BOHR_TO_ANG
    assert np.isclose(z_c_ang - z_top_ang, 2.0, atol=1e-8)


def test_place_adsorbate_side_on_n2_bridge():
    sys, info = slab("Fe", facet=(1, 0, 0), n_layers=5, vacuum=14.0)
    sys2 = place_adsorbate(
        sys,
        "N2",
        info=info,
        site="bridge",
        height=1.8,
        orientation="side-on",
        bond_length=1.20,
    )
    ads = list(sys2.unit_cell)[len(sys.unit_cell) :]
    pos = np.array([list(a.xyz) for a in ads]) * _BOHR_TO_ANG
    # Side-on: bond is along x → both atoms at the same z.
    assert np.isclose(pos[0, 2], pos[1, 2], atol=1e-8)
    # Bond length preserved.
    bl = np.linalg.norm(pos[1] - pos[0])
    assert np.isclose(bl, 1.20, atol=1e-8)


# ---------------------------------------------------------------------------
# New molecule tests
# ---------------------------------------------------------------------------


def test_molecule_no():
    """NO molecule: N at origin, O at +1.151 along z."""
    no = molecule("NO")
    syms = [s for s, _ in no]
    assert syms == ["N", "O"]
    xs = np.array([r for _, r in no])
    # N at origin
    assert np.allclose(xs[0], [0.0, 0.0, 0.0], atol=1e-10)
    # O bonded along +z
    assert np.isclose(np.linalg.norm(xs[1] - xs[0]), 1.151, atol=1e-10)


def test_molecule_nh2():
    """NH2 molecule: N at origin, two H's in bent geometry (103°)."""
    nh2 = molecule("NH2")
    syms = [s for s, _ in nh2]
    assert syms == ["N", "H", "H"]
    xs = np.array([r for _, r in nh2])
    # N at origin
    assert np.allclose(xs[0], [0.0, 0.0, 0.0], atol=1e-10)
    # Both H-N bonds ~1.024 Å
    for i in (1, 2):
        assert np.isclose(np.linalg.norm(xs[i] - xs[0]), 1.024, atol=1e-4)
    # H-N-H angle ~103°
    v1 = xs[1] - xs[0]
    v2 = xs[2] - xs[0]
    dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_deg = np.degrees(np.arccos(dot))
    assert np.isclose(angle_deg, 103.0, atol=1.0)


# ---------------------------------------------------------------------------
# distance kwarg (alias for height)
# ---------------------------------------------------------------------------


def test_place_adsorbate_distance_alias():
    """distance= works identically to height=."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    n_slab = len(sys.unit_cell)
    # Using distance= instead of height=
    sys2 = place_adsorbate(
        sys, "CO", info=info, site="top", distance=2.5, orientation="end-on"
    )
    assert len(sys2.unit_cell) == n_slab + 2
    ads = list(sys2.unit_cell)[n_slab:]
    z_top_ang = max(a.xyz[2] for a in sys.unit_cell) * _BOHR_TO_ANG
    z_c_ang = ads[0].xyz[2] * _BOHR_TO_ANG
    assert np.isclose(z_c_ang - z_top_ang, 2.5, atol=1e-8)


def test_place_adsorbate_height_and_distance_mutex():
    """Both height and distance → ValueError."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    with pytest.raises(ValueError, match="only one"):
        place_adsorbate(sys, "CO", info=info, site="top", height=2.0, distance=2.5)


def test_place_adsorbate_no_height_or_distance():
    """Neither height nor distance → ValueError."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    with pytest.raises(ValueError, match="must be set"):
        place_adsorbate(sys, "CO", info=info, site="top")


# ---------------------------------------------------------------------------
# Orientation modes
# ---------------------------------------------------------------------------


def test_orient_molecule_end_on():
    """end-on: molecule stays along +z, anchor at z=0."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])
    out = _orient_molecule(coords, "end-on")
    assert np.allclose(out[0], [0.0, 0.0, 0.0], atol=1e-10)
    assert np.allclose(out[1], [0.0, 0.0, 1.128], atol=1e-10)


def test_orient_molecule_upright_alias():
    """upright is an alias for end-on."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])
    out = _orient_molecule(coords, "upright")
    assert np.isclose(out[1, 2], 1.128, atol=1e-10)


def test_orient_molecule_side_on():
    """side-on: bond parallel to surface, both atoms at same z."""
    coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]])
    out = _orient_molecule(coords, "side-on")
    assert np.isclose(out[0, 2], out[1, 2], atol=1e-10)
    bl = np.linalg.norm(out[1] - out[0])
    assert np.isclose(bl, 1.10, atol=1e-10)


def test_orient_molecule_tilted():
    """tilted: 45° from surface normal, anchor remains lowest."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])
    out = _orient_molecule(coords, "tilted")
    # Anchor (atom 0) should still be at z=0
    assert np.isclose(out[0, 2], 0.0, atol=1e-10)
    # Bond length preserved
    bl = np.linalg.norm(out[1] - out[0])
    assert np.isclose(bl, 1.128, atol=1e-4)
    # z-component of bond should be ~0.798 (1.128 * cos(45°))
    assert np.isclose(out[1, 2], 1.128 * np.cos(np.radians(45)), atol=1e-4)


def test_orient_molecule_flat_diatomic():
    """flat on a diatomic → side-on behaviour."""
    coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]])
    out = _orient_molecule(coords, "flat")
    assert np.isclose(out[0, 2], out[1, 2], atol=1e-10)


def test_orient_molecule_auto_n2():
    """auto on N2 → side-on."""
    coords = np.array([[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]])
    out = _orient_molecule(coords, "auto", "N2")
    assert np.isclose(out[0, 2], out[1, 2], atol=1e-10)


def test_orient_molecule_auto_co():
    """auto on CO → end-on."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]])
    out = _orient_molecule(coords, "auto", "CO")
    assert np.isclose(out[1, 2], 1.128, atol=1e-10)


def test_orient_molecule_invalid():
    """Invalid orientation → ValueError."""
    coords = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="orientation must be one of"):
        _orient_molecule(coords, "invalid-orientation")


# ---------------------------------------------------------------------------
# Anchor parameter
# ---------------------------------------------------------------------------


def test_anchor_by_index():
    """Anchor by 0-based index — O (idx 1) placed at surface height."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    # Place CO with O as anchor (index 1) so O goes to the surface.
    sys2 = place_adsorbate(
        sys, "CO", info=info, site="top", height=2.0, anchor=1, orientation="end-on"
    )
    n_slab = len(sys.unit_cell)
    ads = list(sys2.unit_cell)[n_slab:]
    # Atom order unchanged: C first, O second (from molecule geometry).
    assert ads[0].Z == 6  # C
    assert ads[1].Z == 8  # O
    # O (anchor) should be at z_top + height.
    z_top_ang = max(a.xyz[2] for a in sys.unit_cell) * _BOHR_TO_ANG
    z_o_ang = ads[1].xyz[2] * _BOHR_TO_ANG
    assert np.isclose(z_o_ang - z_top_ang, 2.0, atol=1e-8)
    # C (built-in geometry has C at z=0, O at z=+1.128); with O as
    # anchor, C shifts below O (negative z in local frame), i.e.
    # buried into the surface — correct for O-down binding.
    z_c_ang = ads[0].xyz[2] * _BOHR_TO_ANG
    assert z_c_ang < z_o_ang


def test_anchor_by_symbol():
    """Anchor by element symbol string — O at surface height."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    sys2 = place_adsorbate(
        sys, "CO", info=info, site="top", height=2.0, anchor="O", orientation="end-on"
    )
    n_slab = len(sys.unit_cell)
    ads = list(sys2.unit_cell)[n_slab:]
    # Atom order unchanged: C first, O second.
    # O (anchor) should be at z_top + height.
    z_top_ang = max(a.xyz[2] for a in sys.unit_cell) * _BOHR_TO_ANG
    z_o_ang = ads[1].xyz[2] * _BOHR_TO_ANG  # O is at index 1
    assert np.isclose(z_o_ang - z_top_ang, 2.0, atol=1e-8)
    # C (built-in geometry has C at z=0, O at z=+1.128); with O as
    # anchor, C shifts below O — correct for O-down binding.
    z_c_ang = ads[0].xyz[2] * _BOHR_TO_ANG
    assert z_c_ang < z_o_ang


def test_anchor_default_is_first_atom():
    """Default anchor is the first atom (heavy atom for built-ins)."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    # Without anchor, CO places C (first atom) at the surface.
    sys2 = place_adsorbate(
        sys, "CO", info=info, site="top", height=2.0, orientation="end-on"
    )
    n_slab = len(sys.unit_cell)
    ads = list(sys2.unit_cell)[n_slab:]
    assert ads[0].Z == 6  # C (default anchor)
    z_top_ang = max(a.xyz[2] for a in sys.unit_cell) * _BOHR_TO_ANG
    z_c_ang = ads[0].xyz[2] * _BOHR_TO_ANG
    assert np.isclose(z_c_ang - z_top_ang, 2.0, atol=1e-8)


def test_anchor_out_of_range():
    """Anchor index out of range → ValueError."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    with pytest.raises(ValueError, match="out of range"):
        place_adsorbate(sys, "CO", info=info, site="top", height=2.0, anchor=5)


def test_anchor_symbol_not_found():
    """Anchor symbol not in molecule → ValueError."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    with pytest.raises(ValueError, match="not found"):
        place_adsorbate(sys, "CO", info=info, site="top", height=2.0, anchor="Fe")


# ---------------------------------------------------------------------------
# _wrap_in_cell
# ---------------------------------------------------------------------------


def test_wrap_in_cell_in_plane():
    """Wrap in-plane component when outside [0, a) × [0, b)."""
    cell = np.eye(3) * 10.0  # 10 Å cube
    pos = np.array([12.0, -3.0, 5.0])
    wrapped = _wrap_in_cell(pos, cell)
    # x: 12 → 2, y: -3 → 7, z unchanged (only in-plane wraps)
    assert np.isclose(wrapped[0], 2.0, atol=1e-10)
    assert np.isclose(wrapped[1], 7.0, atol=1e-10)
    assert np.isclose(wrapped[2], 5.0, atol=1e-10)


def test_wrap_in_cell_z_unchanged():
    """z coordinate is never wrapped."""
    cell = np.eye(3) * 5.0
    pos = np.array([0.0, 0.0, 15.0])
    wrapped = _wrap_in_cell(pos, cell)
    assert np.isclose(wrapped[2], 15.0, atol=1e-10)


def test_wrap_in_cell_already_inside():
    """Position already inside cell is unchanged."""
    cell = np.eye(3) * 10.0
    pos = np.array([3.0, 7.0, 4.0])
    wrapped = _wrap_in_cell(pos, cell)
    assert np.allclose(wrapped, pos, atol=1e-10)


def test_wrap_in_cell_non_orthogonal():
    """Wrapping works with non-orthogonal cell."""
    # Hexagonal cell (e.g., fcc(111) surface cell)
    ax = 2.5
    cell = np.array(
        [[ax, 0.0, 0.0], [ax / 2, ax * np.sqrt(3) / 2, 0.0], [0.0, 0.0, 20.0]],
        dtype=float,
    )
    pos = np.array([ax * 1.3, ax * 0.8, 10.0])
    wrapped = _wrap_in_cell(pos, cell)
    # Should be back in [0, a) × [0, b)
    frac = np.linalg.solve(cell.T, wrapped)
    assert 0.0 <= frac[0] < 1.0
    assert 0.0 <= frac[1] < 1.0


# ---------------------------------------------------------------------------
# Default orientation "auto" end-to-end
# ---------------------------------------------------------------------------


def test_auto_orientation_co():
    """CO at top site with auto orientation → C-down end-on."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    sys2 = place_adsorbate(sys, "CO", info=info, site="top", height=2.0)
    n_slab = len(sys.unit_cell)
    ads = list(sys2.unit_cell)[n_slab:]
    assert ads[0].Z == 6  # C (anchor)
    assert ads[1].Z == 8  # O above


def test_auto_orientation_n2():
    """N2 at bridge with auto orientation → side-on."""
    sys, info = slab("Fe", facet=(1, 0, 0), n_layers=5, vacuum=14.0)
    sys2 = place_adsorbate(
        sys, "N2", info=info, site="bridge", height=1.8, bond_length=1.20
    )
    ads = list(sys2.unit_cell)[len(sys.unit_cell) :]
    pos = np.array([list(a.xyz) for a in ads]) * _BOHR_TO_ANG
    # Side-on → both N at same z
    assert np.isclose(pos[0, 2], pos[1, 2], atol=1e-8)
    bl = np.linalg.norm(pos[1] - pos[0])
    assert np.isclose(bl, 1.20, atol=1e-8)


def test_auto_orientation_nh3():
    """NH3 at top site with auto orientation → N-down end-on."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    sys2 = place_adsorbate(sys, "NH3", info=info, site="top", height=2.0)
    n_slab = len(sys.unit_cell)
    ads = list(sys2.unit_cell)[n_slab:]
    assert ads[0].Z == 7  # N (anchor), lowest atom
    # All H's are above N
    z_n = ads[0].xyz[2]
    for a in ads[1:]:
        assert a.xyz[2] >= z_n


# ---------------------------------------------------------------------------
# Edge cases: single atom, explicit geometry
# ---------------------------------------------------------------------------


def test_place_atom():
    """Place a single H atom at a top site."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    n_slab = len(sys.unit_cell)
    sys2 = place_adsorbate(sys, "H", info=info, site="top", height=1.5)
    assert len(sys2.unit_cell) == n_slab + 1
    ads = list(sys2.unit_cell)[n_slab:]
    assert ads[0].Z == 1
    z_top = max(a.xyz[2] for a in sys.unit_cell) * _BOHR_TO_ANG
    z_h = ads[0].xyz[2] * _BOHR_TO_ANG
    assert np.isclose(z_h - z_top, 1.5, atol=1e-8)


def test_place_explicit_geometry():
    """Place an explicit geometry tuple."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    n_slab = len(sys.unit_cell)
    # Custom OH radical via explicit tuple
    oh = (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.970)))
    sys2 = place_adsorbate(
        sys, oh, info=info, site="top", height=2.0, orientation="end-on"
    )
    assert len(sys2.unit_cell) == n_slab + 2
    ads = list(sys2.unit_cell)[n_slab:]
    assert ads[0].Z == 8  # O (first atom in tuple, default anchor)
    z_top = max(a.xyz[2] for a in sys.unit_cell) * _BOHR_TO_ANG
    z_o = ads[0].xyz[2] * _BOHR_TO_ANG
    assert np.isclose(z_o - z_top, 2.0, atol=1e-8)


def test_place_at_explicit_position():
    """Place at an explicit (x, y) position."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    sys2 = place_adsorbate(sys, "H", position=(1.0, 2.0), height=2.0)
    n_slab = len(sys.unit_cell)
    ads = list(sys2.unit_cell)[n_slab:]
    pos = np.array([list(a.xyz) for a in ads]) * _BOHR_TO_ANG
    assert np.isclose(pos[0, 0], 1.0, atol=1e-6)
    assert np.isclose(pos[0, 1], 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# ValueError error messages
# ---------------------------------------------------------------------------


def test_unknown_molecule_name():
    """Unknown molecule name → KeyError."""
    with pytest.raises(KeyError, match="not in the built-in library"):
        molecule("UNKNOWN")


def test_unknown_site_label():
    """Unknown site label → ValueError."""
    sys, info = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)
    with pytest.raises(ValueError, match="not recognised"):
        place_adsorbate(sys, "H", info=info, site="not-a-site", height=2.0)


def test_no_info_no_position():
    """Without info and without position → ValueError."""
    sys = slab("Cu", facet=(1, 1, 1), n_layers=4, vacuum=14.0)[0]
    with pytest.raises(ValueError, match="pass info="):
        place_adsorbate(sys, "H", site="top", height=2.0)
