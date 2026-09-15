"""Phase K6 — generalized regular k-point grids."""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.kpoints as kpoints_mod


ANGSTROM_TO_BOHR = 1.8897261339213


def _skew_3d() -> vq.PeriodicSystem:
    lattice = (
        np.array(
            [
                [4.1, 0.0, 0.0],
                [0.8, 5.2, 0.0],
                [0.4, 1.1, 6.3],
            ],
            dtype=float,
        )
        * ANGSTROM_TO_BOHR
    )
    return vq.PeriodicSystem(3, lattice, [vq.Atom(14, [0.0, 0.0, 0.0])])


def _min_distance(system: vq.PeriodicSystem, kp: vq.KPoints) -> float:
    frac = np.asarray(kp.kpoints_frac, dtype=float)
    wrapped = frac - np.round(frac)
    nonzero = wrapped[np.linalg.norm(wrapped, axis=1) > 1.0e-12]
    if nonzero.size == 0:
        return np.inf
    cart = (np.asarray(system.reciprocal_lattice()) @ nonzero.T).T
    return float(np.linalg.norm(cart, axis=1).min())


def _frac_set(kpoints: np.ndarray) -> set[tuple[float, float, float]]:
    frac = np.asarray(kpoints, dtype=float) % 1.0
    frac[np.isclose(frac, 1.0, atol=1.0e-10)] = 0.0
    frac[np.isclose(frac, 0.0, atol=1.0e-10)] = 0.0
    return {tuple(np.round(k, 12)) for k in frac}


def test_generalized_regular_is_public():
    assert hasattr(vq.KPoints, "generalized_regular")
    assert hasattr(vq.KPoints, "optimal")
    assert hasattr(vq.KPoints, "from_database")


def test_diagonal_hnf_matches_gamma_centred_mesh():
    sys = _skew_3d()

    gr = vq.KPoints.generalized_regular(sys, np.diag([2, 2, 2]))
    mp = vq.KPoints.gamma_centred(sys, [2, 2, 2])

    assert gr.kind == "generalized-regular"
    assert gr.grid_matrix is not None
    assert int(round(np.linalg.det(gr.grid_matrix))) == 8
    assert len(gr) == 8
    assert gr.weights.sum() == pytest.approx(1.0)

    assert _frac_set(gr.kpoints_frac) == _frac_set(mp.kpoints_frac)


def test_optimal_returns_exact_target_count_and_bloch_roundtrip():
    sys = _skew_3d()

    kp = vq.KPoints.optimal(sys, 8)
    bm = kp.to_bloch_kmesh()

    assert kp.kind == "generalized-regular"
    assert len(kp) == 8
    assert kp.grid_matrix is not None
    assert int(round(np.linalg.det(kp.grid_matrix))) == 8
    assert len(bm) == len(kp)
    np.testing.assert_allclose(kp.weights.sum(), 1.0, atol=1e-12)
    assert any(np.allclose(k, 0.0, atol=1e-12) for k in kp.kpoints_frac)


def test_optimal_is_at_least_as_uniform_as_diagonal_candidates():
    sys = _skew_3d()

    optimal = vq.KPoints.optimal(sys, 8)
    diagonal_candidates = [
        vq.KPoints.generalized_regular(sys, np.diag(mesh))
        for mesh in ([1, 1, 8], [1, 2, 4], [1, 4, 2], [2, 2, 2])
    ]

    best_diagonal = max(_min_distance(sys, kp) for kp in diagonal_candidates)
    assert _min_distance(sys, optimal) >= best_diagonal - 1.0e-12


def test_optimal_metallic_multiplies_target_count():
    sys = _skew_3d()

    kp = vq.KPoints.optimal(sys, 3, metallic=True)

    assert len(kp) == 12
    assert int(round(np.linalg.det(kp.grid_matrix))) == 12


def test_generalized_regular_rejects_non_hnf_matrix():
    sys = _skew_3d()

    with pytest.raises(ValueError, match="upper-triangular HNF"):
        vq.KPoints.generalized_regular(sys, [[1, 0, 0], [1, 2, 0], [0, 0, 2]])


def test_optimal_rejects_low_dimensional_system():
    sys = vq.PeriodicSystem(
        2,
        np.diag([4.0, 4.0, 30.0]) * ANGSTROM_TO_BOHR,
        [vq.Atom(6, [0.0, 0.0, 0.0])],
    )

    with pytest.raises(ValueError, match="3D"):
        vq.KPoints.optimal(sys, 8)


def test_optimal_rejects_nonpositive_target():
    sys = _skew_3d()

    with pytest.raises(ValueError, match="positive"):
        vq.KPoints.optimal(sys, 0)


def test_database_lookup_parses_weighted_remote_table(monkeypatch):
    sys = _skew_3d()
    calls = []

    def fake_download(url: str, *, timeout: float) -> str:
        calls.append((url, timeout))
        return """
        3
          .000   .000   .000   .125       (Gamma)
          .500   .000   .000   .375       (X)
          .500   .500   .000   .500       (M)
        """

    monkeypatch.setattr(
        kpoints_mod,
        "_download_kpoint_database_table",
        fake_download,
    )

    kp = vq.KPoints.from_database(
        sys,
        "simple cubic",
        4,
        base_url="http://example.invalid/kpts",
        timeout=2.5,
    )
    bm = kp.to_bloch_kmesh()

    assert calls == [("http://example.invalid/kpts/sc/regular.04", 2.5)]
    assert kp.kind == "database"
    assert len(kp) == 3
    np.testing.assert_allclose(
        kp.kpoints_frac,
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(kp.weights, [0.125, 0.375, 0.5])
    np.testing.assert_allclose(np.asarray(bm.weights), kp.weights)


def test_database_lookup_builds_special_and_hex_urls(monkeypatch):
    sys = _skew_3d()
    urls = []

    def fake_download(url: str, *, timeout: float) -> str:
        urls.append(url)
        return "1\n0.0 0.0 0.0 1.0\n"

    monkeypatch.setattr(
        kpoints_mod,
        "_download_kpoint_database_table",
        fake_download,
    )

    vq.KPoints.from_database(
        sys,
        "fcc",
        [6],
        special=True,
        base_url="http://example.invalid/kpts/",
    )
    vq.KPoints.from_database(
        sys,
        "hexagonal",
        (6, 3),
        base_url="http://example.invalid/kpts/",
    )

    assert urls == [
        "http://example.invalid/kpts/fcc/special.06",
        "http://example.invalid/kpts/hex/regular.6.3",
    ]


def test_database_lookup_rejects_bad_table_count(monkeypatch):
    sys = _skew_3d()

    monkeypatch.setattr(
        kpoints_mod,
        "_download_kpoint_database_table",
        lambda url, *, timeout: "2\n0.0 0.0 0.0 1.0\n",
    )

    with pytest.raises(ValueError, match="count mismatch"):
        vq.KPoints.from_database(sys, "sc", 1)


def test_database_lookup_rejects_hex_special_mesh():
    sys = _skew_3d()

    with pytest.raises(ValueError, match="hexagonal database"):
        vq.KPoints.from_database(sys, "hex", (6, 3), special=True)


def test_database_lookup_rejects_low_dimensional_system():
    sys = vq.PeriodicSystem(
        2,
        np.diag([4.0, 4.0, 30.0]) * ANGSTROM_TO_BOHR,
        [vq.Atom(6, [0.0, 0.0, 0.0])],
    )

    with pytest.raises(ValueError, match="from_database.*3D"):
        vq.KPoints.from_database(sys, "sc", 1)
