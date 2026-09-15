"""Phase K3 — band-path k-points via seekpath (HPKOT) auto-detection.

Pinned contracts:

1. **API surface** — ``KPoints.band_path(system)`` is public; default
   ``scheme="auto"`` uses seekpath HPKOT auto-detection. ``"manual"``
   routes through the existing ``bands.kpath_from_segments``.

2. **HPKOT canonical paths** — for the standard Bravais lattices,
   the labels match the published Hinuma 2017 conventions:

   - **FCC (Fm-3m, SG 225)**: Γ–X–U|K–Γ–L–W–X
   - **BCC (Im-3m, SG 229)**: Γ–H–N–Γ–P–H|P–N
   - **SC  (Pm-3m, SG 221)**: Γ–X–M–Γ–R–X|R–M
   - **HEX (P6_3/mmc, SG 194)**: Γ–M–K–Γ–A–L–H–A|L–M|H–K
   - **TET (P4/mmm, SG 123)**: Γ–X–M–Γ–Z–R–A–Z|X–R|M–A

3. **Label prettification** — seekpath's ``"GAMMA"`` is converted to
   the unicode ``"Γ"`` for plot tick rendering.

4. **Round-trip integrity** — ``KPoints.band_path(...).to_kpath()``
   produces a valid :class:`vibeqc.bands.KPath` whose
   ``kpoints_cart`` shape matches and whose ``distances`` are
   monotonically non-decreasing.

5. **End-to-end with band_structure_hcore** — handing the K3 KPath
   to :func:`vibeqc.band_structure_hcore` produces a valid
   :class:`BandStructure` of the right shape with a sensible
   Fermi-level recovery for closed-shell systems.

6. **Manual scheme** — ``scheme="manual"`` requires segments and
   raises a clean error otherwise.

7. **Reference-distance control** — smaller ``reference_distance``
   produces denser path discretisation.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# Cell builders (vibeqc convention: columns = lattice vectors)
# ---------------------------------------------------------------------------

def _fcc(a_ang: float = 5.43, Z: int = 14) -> vq.PeriodicSystem:
    a = a_ang * ANGSTROM_TO_BOHR
    lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
    return vq.PeriodicSystem(3, lat, [vq.Atom(Z, [0.0, 0.0, 0.0])])


def _bcc(a_ang: float = 3.16, Z: int = 74) -> vq.PeriodicSystem:
    a = a_ang * ANGSTROM_TO_BOHR
    lat = np.array([[-a/2, a/2, a/2], [a/2, -a/2, a/2], [a/2, a/2, -a/2]])
    return vq.PeriodicSystem(3, lat, [vq.Atom(Z, [0.0, 0.0, 0.0])])


def _sc(a_ang: float = 3.36, Z: int = 84) -> vq.PeriodicSystem:
    a = a_ang * ANGSTROM_TO_BOHR
    return vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(Z, [0.0, 0.0, 0.0])])


def _hex_be() -> vq.PeriodicSystem:
    a, c = 2.29 * ANGSTROM_TO_BOHR, 3.58 * ANGSTROM_TO_BOHR
    lat = np.array([[a,    -a/2,             0.0],
                     [0.0,  a*np.sqrt(3)/2,  0.0],
                     [0.0,  0.0,             c]])
    p1 = np.zeros(3)
    p2 = lat @ np.array([1/3, 2/3, 1/2])
    return vq.PeriodicSystem(3, lat, [
        vq.Atom(4, p1.tolist()), vq.Atom(4, p2.tolist())])


def _tet(a_ang: float = 3.0, c_ang: float = 5.0) -> vq.PeriodicSystem:
    a = a_ang * ANGSTROM_TO_BOHR
    c = c_ang * ANGSTROM_TO_BOHR
    return vq.PeriodicSystem(3, np.diag([a, a, c]), [vq.Atom(14, [0, 0, 0])])


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_band_path_is_public():
    assert hasattr(vq.KPoints, "band_path")
    assert hasattr(vq.KPoints, "to_kpath")


def test_default_scheme_is_auto():
    sys = _fcc()
    kp = vq.KPoints.band_path(sys)
    assert kp.kind == "band-path"
    assert isinstance(kp, vq.KPoints)


# ---------------------------------------------------------------------------
# 2. HPKOT canonical paths — sanity-check labels against Hinuma 2017
# ---------------------------------------------------------------------------

def _path_labels(kp: vq.KPoints) -> list:
    return [lbl for _, lbl in kp.labels]


def test_fcc_hpkot_path_labels():
    sys = _fcc()
    kp = vq.KPoints.band_path(sys)
    labels = _path_labels(kp)
    # Hinuma 2017 cF path: Γ → X → U|K → Γ → L → W → X
    assert labels == ["Γ", "X", "U|K", "Γ", "L", "W", "X"]


def test_bcc_hpkot_path_labels():
    sys = _bcc()
    kp = vq.KPoints.band_path(sys)
    labels = _path_labels(kp)
    # Hinuma 2017 cI path: Γ → H → N → Γ → P → H|P → N
    assert labels == ["Γ", "H", "N", "Γ", "P", "H|P", "N"]


def test_sc_hpkot_path_labels():
    sys = _sc()
    kp = vq.KPoints.band_path(sys)
    labels = _path_labels(kp)
    # Hinuma 2017 cP path: Γ → X → M → Γ → R → X|R → M
    assert labels == ["Γ", "X", "M", "Γ", "R", "X|R", "M"]


def test_hex_hpkot_path_labels():
    sys = _hex_be()
    kp = vq.KPoints.band_path(sys)
    labels = _path_labels(kp)
    # Hinuma 2017 hP path: Γ → M → K → Γ → A → L → H → A|L → M|H → K
    assert labels == ["Γ", "M", "K", "Γ", "A", "L", "H", "A|L", "M|H", "K"]


def test_tetragonal_hpkot_path_labels():
    sys = _tet()
    kp = vq.KPoints.band_path(sys)
    labels = _path_labels(kp)
    # Hinuma 2017 tP path: Γ → X → M → Γ → Z → R → A → Z|X → R|M → A
    assert labels == ["Γ", "X", "M", "Γ", "Z", "R", "A", "Z|X", "R|M", "A"]


# ---------------------------------------------------------------------------
# 3. Label prettification (GAMMA → Γ)
# ---------------------------------------------------------------------------

def test_gamma_is_unicode():
    sys = _fcc()
    kp = vq.KPoints.band_path(sys)
    assert "Γ" in [lbl for _, lbl in kp.labels]
    assert "GAMMA" not in [lbl for _, lbl in kp.labels]


# ---------------------------------------------------------------------------
# 4. KPath round-trip
# ---------------------------------------------------------------------------

def test_to_kpath_round_trip_shapes():
    sys = _fcc()
    kp = vq.KPoints.band_path(sys)
    kpath = kp.to_kpath()
    assert kpath.kpoints_cart.shape == kp.kpoints_cart.shape
    assert kpath.kpoints_frac.shape == kp.kpoints_frac.shape
    assert kpath.distances.shape == (len(kp),)
    assert kpath.distances[0] == 0.0
    # distances monotonically non-decreasing
    diffs = np.diff(kpath.distances)
    assert (diffs >= -1e-12).all()


def test_to_kpath_only_for_band_path_kind():
    sys = _fcc()
    kp_mp = vq.KPoints.monkhorst_pack(sys, [2, 2, 2])
    with pytest.raises(ValueError, match="band-path"):
        kp_mp.to_kpath()


# ---------------------------------------------------------------------------
# 5. End-to-end with band_structure_hcore
# ---------------------------------------------------------------------------

def test_band_structure_hcore_consumes_k3_kpath():
    """K3 KPath produces a valid BandStructure on diamond Si."""
    a = 5.43 * ANGSTROM_TO_BOHR
    lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
    p1 = np.zeros(3)
    p2 = 0.25 * (lat[:, 0] + lat[:, 1] + lat[:, 2])
    sys = vq.PeriodicSystem(3, lat, [
        vq.Atom(14, p1.tolist()), vq.Atom(14, p2.tolist())])
    kp = vq.KPoints.band_path(sys)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    kpath = kp.to_kpath()
    bands = vq.band_structure_hcore(sys, basis, kpath,
                                      n_electrons_per_cell=8)
    assert bands.energies.shape == (len(kp), basis.nbasis)
    assert bands.e_fermi is not None
    # Bands at each k are sorted ascending.
    for i in range(len(kp)):
        assert (np.diff(bands.energies[i]) >= -1e-9).all()


# ---------------------------------------------------------------------------
# 6. Manual scheme
# ---------------------------------------------------------------------------

def test_manual_scheme_requires_segments():
    sys = _sc()
    with pytest.raises(ValueError, match="segments"):
        vq.KPoints.band_path(sys, scheme="manual")


def test_manual_scheme_passes_segments_through():
    sys = _sc()
    segments = [
        ((0, 0, 0), "Γ", (0.5, 0, 0), "X"),
        ((0.5, 0, 0), "X", (0.5, 0.5, 0), "M"),
    ]
    kp = vq.KPoints.band_path(sys, scheme="manual", segments=segments,
                                points_per_segment=10)
    assert kp.kind == "band-path"
    assert len(kp) == 21  # 11 first + 10 second (start point shared)
    assert [lbl for _, lbl in kp.labels] == ["Γ", "X", "M"]


def test_unknown_scheme_raises():
    sys = _sc()
    with pytest.raises(ValueError, match="scheme"):
        vq.KPoints.band_path(sys, scheme="nonsense")


# ---------------------------------------------------------------------------
# 7. Reference-distance density control
# ---------------------------------------------------------------------------

def test_reference_distance_controls_density():
    sys = _fcc()
    sparse = vq.KPoints.band_path(sys, reference_distance=0.05)
    dense = vq.KPoints.band_path(sys, reference_distance=0.005)
    assert len(dense) > len(sparse)
    # Same path → same labels
    assert _path_labels(sparse) == _path_labels(dense)


# ---------------------------------------------------------------------------
# 8. Weights sum to 1 (type-consistency with MP/IBZ KPoints)
# ---------------------------------------------------------------------------

def test_band_path_weights_sum_to_one():
    sys = _fcc()
    kp = vq.KPoints.band_path(sys)
    np.testing.assert_allclose(kp.weights.sum(), 1.0, atol=1e-12)
    # All weights equal (band path has no integration meaning).
    np.testing.assert_allclose(kp.weights, kp.weights[0], atol=1e-12)
