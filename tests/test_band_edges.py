"""One band-edge classifier for every periodic surface (GitLab #176).

Until #176 the same spectrum was classified three ways: the ``.out``
band-extrema block and the native semiempirical k-route fields read the
occupations state by state, :func:`vibeqc.frontier_bands` read a band index,
and each open-shell driver kept its own zero-temperature fill. On the two-k
band-overlap discriminator ``eps[k0] = [-0.20, 0.30, 0.90]``,
``eps[k1] = [-0.60, -0.50, 0.80]`` (two electrons per cell, equal weights):

* ``.out``: ``gap (indirect) = 0.300000 Ha (8.163 eV)`` -- an insulator;
* ``frontier_bands(result, n_occ=1)``: ``gap = -0.3``;
* open-shell T = 0: per-k occupations ``[[1,0,0],[1,0,0]]`` with the global
  ``mu = -0.35`` computed and not used.

These gates pin the shared contract (:mod:`vibeqc.band_edges`): the Fermi
level crossing a band is detected, measured as the band overlap at the Fermi
filling, flagged ``is_metallic``, and reported identically by every surface.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.band_edges import (
    OCCUPATION_TOLERANCE,
    BandChannel,
    kpoint_band_edges,
    result_band_edges,
)
from vibeqc.output.formats.band_edges import (
    band_edge_manifest_fields,
    band_extrema_block,
)
from vibeqc.smearing import (
    apply_smearing,
    apply_smearing_open_shell,
    spin_channel_occupations,
)

_EPS = [np.array([-0.20, 0.30, 0.90]), np.array([-0.60, -0.50, 0.80])]
_W = [0.5, 0.5]


def _channel(eps, occ, g=2.0, spin=None, rohf=False):
    return BandChannel(spin, tuple(eps), tuple(occ), g, rohf)


def _state_class_edges(eps_per_k, occ_per_k, g=2.0):
    """The #426 state-class reading, re-derived independently."""
    vbm = max(
        float(e) for eps, occ in zip(eps_per_k, occ_per_k)
        for e, o in zip(eps, occ) if o > OCCUPATION_TOLERANCE
    )
    cbm = min(
        float(e) for eps, occ in zip(eps_per_k, occ_per_k)
        for e, o in zip(eps, occ) if o < g - OCCUPATION_TOLERANCE
    )
    return vbm, cbm, cbm - vbm


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


def test_band_crossing_metal_reports_the_band_overlap_not_the_level_spacing():
    occ = apply_smearing(
        _EPS, weights=_W, n_electrons_per_cell=2.0, n_occ_each=1
    ).occupations_per_k
    edges = kpoint_band_edges([_channel(_EPS, occ)], weights=_W)

    assert edges.fermi_level_crosses_band
    assert not edges.fractional_fermi_group
    assert edges.is_metallic
    assert edges.classification == "metal"
    # Band pair (0, 1) at the Fermi filling n_f = 1: band 0 tops out at k0,
    # band 1 bottoms out at k1.
    assert edges.valence_band_max == pytest.approx(-0.20)
    assert edges.valence_band_max_k == 0
    assert edges.conduction_band_min == pytest.approx(-0.50)
    assert edges.conduction_band_min_k == 1
    assert edges.indirect_gap == pytest.approx(-0.30)
    assert edges.direct_gap == pytest.approx(0.10)
    assert edges.direct_gap_k == 1
    # The positive spacing the pre-#176 reporter printed as the gap is kept
    # only under its own name.
    assert edges.gap_above_fermi_manifold == pytest.approx(0.30)
    assert _state_class_edges(_EPS, occ)[2] == pytest.approx(0.30)


def test_per_k_and_global_fills_of_one_metal_report_the_same_overlap():
    per_k = [np.array([2.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
    global_ = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 2.0, 0.0])]
    a = kpoint_band_edges([_channel(_EPS, per_k)], weights=_W)
    b = kpoint_band_edges([_channel(_EPS, global_)], weights=_W)
    assert a.indirect_gap == pytest.approx(b.indirect_gap)
    assert a.is_metallic and b.is_metallic
    # The per-k fill is not a crossing: its counts agree, the classes
    # themselves are inverted, and that is what the negative gap measures.
    assert not a.fermi_level_crosses_band
    assert b.fermi_level_crosses_band


def test_gapped_mesh_edges_are_the_state_class_edges_bit_for_bit():
    rng = np.random.default_rng(176)
    for _ in range(50):
        n_k, n_b, n_occ = 5, 6, 3
        eps = [
            np.sort(np.concatenate([
                rng.uniform(-1.0, -0.5, n_occ), rng.uniform(0.1, 1.0, n_b - n_occ)
            ]))
            for _ in range(n_k)
        ]
        w = np.full(n_k, 1.0 / n_k)
        occ = apply_smearing(
            eps, weights=w, n_electrons_per_cell=2.0 * n_occ, n_occ_each=n_occ
        ).occupations_per_k
        edges = kpoint_band_edges([_channel(eps, occ)], weights=w)
        vbm, cbm, gap = _state_class_edges(eps, occ)
        assert not edges.is_metallic
        assert edges.classification == "insulator"
        assert edges.valence_band_max == vbm
        assert edges.conduction_band_min == cbm
        assert edges.indirect_gap == gap
        assert edges.gap_above_fermi_manifold == gap


def test_every_band_crossing_global_fill_measures_a_strictly_negative_gap():
    """The module docstring's proof, exercised: integer n, counts differ =>
    eps[k1][n] < mu < eps[k2][n - 1], so the band-pair gap is < 0."""
    rng = np.random.default_rng(2026)
    crossings = 0
    for _ in range(400):
        n_k, n_b = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        n_occ = int(rng.integers(1, n_b))
        eps = [np.sort(rng.normal(size=n_b)) for _ in range(n_k)]
        # Equal weights put the electron count on a state boundary, so the
        # fill is integer (unequal ones generically leave one fractional
        # boundary state, the fractional_fermi_group case).
        w = np.full(n_k, 1.0 / n_k)
        res = apply_smearing(
            eps, weights=w, n_electrons_per_cell=2.0 * n_occ, n_occ_each=n_occ
        )
        edges = kpoint_band_edges(
            [_channel(eps, res.occupations_per_k)], weights=w
        )
        if edges.fermi_level_crosses_band:
            crossings += 1
            assert edges.indirect_gap < 0.0
            assert edges.is_metallic
            # frontier_bands' band-index reading agrees on every crossing.
            r = types.SimpleNamespace(mo_energies=eps)
            assert vq.frontier_bands(r, n_occ=n_occ).gap == pytest.approx(
                edges.indirect_gap, abs=0.0
            )
        elif not edges.fractional_fermi_group:
            assert edges.indirect_gap >= 0.0
    assert crossings > 50  # the property was actually exercised


def test_half_filled_band_measures_minus_its_width():
    """A non-integer Fermi filling names ONE partially filled band."""
    eps = [np.array([-0.40, 0.60]), np.array([-0.10, 0.70])]
    # One electron per cell, restricted: band 0 is half filled.
    occ = [np.array([2.0, 0.0]), np.array([0.0, 0.0])]
    edges = kpoint_band_edges([_channel(eps, occ)], weights=_W)
    assert edges.fermi_level_crosses_band
    assert edges.indirect_gap == pytest.approx(-0.30)  # -(width of band 0)
    assert edges.valence_band_max == pytest.approx(-0.10)
    assert edges.conduction_band_min == pytest.approx(-0.40)


def test_fractional_fermi_group_keeps_the_state_class_measure():
    eps = [np.array([-1.0, 0.0]), np.array([0.0, 1.0])]
    occ = [np.array([2.0, 1.0]), np.array([1.0, 0.0])]
    edges = kpoint_band_edges([_channel(eps, occ)], weights=_W)
    assert edges.fractional_fermi_group
    assert not edges.fermi_level_crosses_band
    assert edges.is_metallic
    assert edges.indirect_gap == pytest.approx(0.0)


def test_empty_classes_are_nan_never_a_metal():
    eps = [np.array([-1.0, -0.5]), np.array([-0.9, -0.4])]
    full = kpoint_band_edges([_channel(eps, [np.full(2, 2.0)] * 2)])
    assert math.isnan(full.conduction_band_min)
    assert math.isnan(full.indirect_gap)
    assert full.classification == "not-measured"
    assert not full.is_metallic


def test_open_shell_edges_combine_both_spin_channels():
    eps_a = [np.array([-0.5, 0.2]), np.array([-0.4, 0.3])]
    eps_b = [np.array([-0.3, 0.1]), np.array([-0.2, 0.4])]
    occ_a = [np.array([1.0, 0.0])] * 2
    occ_b = [np.array([0.0, 0.0])] * 2
    edges = kpoint_band_edges(
        [_channel(eps_a, occ_a, 1.0, "alpha"), _channel(eps_b, occ_b, 1.0, "beta")]
    )
    assert [c.spin for c in edges.channels] == ["alpha", "beta"]
    # Combined: VBM from alpha (-0.4 at k1), CBM from beta (-0.3 at k0).
    assert edges.valence_band_max == pytest.approx(-0.4)
    assert edges.conduction_band_min == pytest.approx(-0.3)
    assert edges.indirect_gap == pytest.approx(0.1)
    assert not edges.is_metallic


def test_rohf_singly_occupied_orbitals_are_valence_not_fractional():
    eps = [np.array([-0.8, -0.2, 0.5]), np.array([-0.7, -0.1, 0.6])]
    occ = [np.array([2.0, 1.0, 0.0])] * 2
    edges = kpoint_band_edges([_channel(eps, occ, 2.0, None, rohf=True)])
    assert not edges.fractional_fermi_group
    assert edges.valence_band_max == pytest.approx(-0.1)
    assert edges.conduction_band_min == pytest.approx(0.5)
    assert edges.indirect_gap == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# One producer: the open-shell zero-temperature fill
# ---------------------------------------------------------------------------


def test_open_shell_t_zero_uses_one_fermi_level_per_spin():
    a, b = apply_smearing_open_shell(_EPS, _EPS, weights=_W, n_alpha=1.0, n_beta=1.0)
    for res in (a, b):
        np.testing.assert_array_equal(res.occupations_per_k[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(res.occupations_per_k[1], [1.0, 1.0, 0.0])
        assert res.mu == pytest.approx(-0.35)
        assert res.entropy == 0.0


def test_open_shell_t_zero_gapped_mesh_is_the_per_k_aufbau():
    eps = [np.array([-0.5, -0.3, 0.1, 0.2]), np.array([-0.4, -0.2, 0.15, 0.25])]
    res = spin_channel_occupations(eps, weights=_W, n_electrons=2.0)
    for occ in res.occupations_per_k:
        np.testing.assert_array_equal(occ, [1.0, 1.0, 0.0, 0.0])
    assert res.mu == pytest.approx(0.5 * (-0.2 + 0.1))


def test_open_shell_mom_holds_its_permuted_block_positionally():
    permuted = [np.array([0.30, -0.20, 0.90]), np.array([-0.60, -0.50, 0.80])]
    res = spin_channel_occupations(
        permuted, weights=_W, n_electrons=1.0, fixed_occupied_subspace=True
    )
    for occ in res.occupations_per_k:
        np.testing.assert_array_equal(occ, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero-temperature contract"):
        spin_channel_occupations(
            permuted, weights=_W, n_electrons=1.0, smearing=0.01,
            fixed_occupied_subspace=True,
        )


def test_open_shell_single_spectrum_keeps_the_historical_aufbau():
    eps = [np.array([-0.5, -0.5, 0.2])]
    res = spin_channel_occupations(eps, weights=[1.0], n_electrons=1.0)
    np.testing.assert_array_equal(res.occupations_per_k[0], [1.0, 0.0, 0.0])


def test_bipole_open_shell_helper_reports_the_real_chemical_potential():
    from vibeqc.pbc_bipole_common import unrestricted_occupations_per_spin

    occ, mu, entropy = unrestricted_occupations_per_spin(
        _EPS, 1, smearing_T=0.0, weights=_W
    )
    np.testing.assert_array_equal(occ[1], [1.0, 1.0, 0.0])
    assert mu == pytest.approx(-0.35)  # was 0.0 before #176
    assert entropy == 0.0


# ---------------------------------------------------------------------------
# Every surface reads the same record
# ---------------------------------------------------------------------------


def _metal_result(**extra):
    occ = apply_smearing(
        _EPS, weights=_W, n_electrons_per_cell=2.0, n_occ_each=1
    ).occupations_per_k
    return types.SimpleNamespace(
        mo_energies=list(_EPS), occupations=list(occ), kpoint_weights=_W,
        converged=True, **extra,
    )


def test_reporter_accessor_and_manifest_agree_on_the_discriminator():
    from vibeqc import periodic_runner as pr

    result = _metal_result()
    edges = result_band_edges(result)
    frontier = vq.frontier_bands(result)
    fields = band_edge_manifest_fields(edges)
    text = pr._band_summary(result)

    assert frontier.gap == edges.indirect_gap == fields["indirect_gap"]
    assert frontier.is_metallic and fields["is_metallic"]
    assert set(fields) == {
        "indirect_gap", "direct_gap", "valence_band_max",
        "conduction_band_min", "gap_above_fermi_manifold", "is_metallic",
    }  # the semiempirical k-route key set (#426)
    assert "gap (indirect) =  -0.300000 Ha" in text
    assert "occupation class = metallic" in text
    # frontier_bands' explicit band-index reading agrees too.
    assert vq.frontier_bands(result, n_occ=1).gap == pytest.approx(frontier.gap)


def test_manifest_fields_mark_absent_edges_not_measured():
    eps = [np.array([-1.0, -0.5]), np.array([-0.9, -0.4])]
    edges = kpoint_band_edges([_channel(eps, [np.full(2, 2.0)] * 2)])
    fields = band_edge_manifest_fields(edges)
    assert fields["conduction_band_min"] == "not-measured"
    assert fields["indirect_gap"] == "not-measured"
    assert fields["is_metallic"] is False


def test_unrestricted_accessor_needs_a_spin_or_the_classifier():
    occ_a = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    r = types.SimpleNamespace(
        mo_energies_alpha=list(_EPS), mo_energies_beta=list(_EPS),
        occupations_alpha=occ_a, occupations_beta=occ_a,
    )
    with pytest.raises(TypeError, match="pass spin="):
        vq.per_k_band_energies(r)
    assert len(vq.per_k_band_energies(r, spin="beta")) == 2
    combined = vq.frontier_bands(r)
    assert combined.edges is not None
    assert [c.spin for c in combined.edges.channels] == ["alpha", "beta"]
    assert vq.frontier_bands(r, spin="alpha").gap == pytest.approx(
        combined.edges.channels[0].indirect_gap
    )


def test_band_extrema_block_is_silent_for_gamma_and_unreadable_results():
    assert band_extrema_block(None) == ""
    one_k = kpoint_band_edges([_channel([_EPS[0]], [np.array([2.0, 0.0, 0.0])])])
    assert band_extrema_block(one_k) == ""


def test_dos_weights_follow_the_global_fill():
    from vibeqc.periodic_gapw_postscf import compute_dos_from_result

    kmesh = types.SimpleNamespace(weights=_W)
    result = types.SimpleNamespace(mo_energies_k=list(_EPS), kmesh=kmesh)
    grid = np.linspace(-1.0, 1.2, 4001)
    _, dos = compute_dos_from_result(result, n_elec=2, e_grid=grid, sigma=0.005)
    de = grid[1] - grid[0]
    # The occupied DOS integrates to the electron count ...
    assert float(np.sum(dos) * de) == pytest.approx(2.0, abs=1.0e-3)
    # ... and carries no weight at the empty k0 state (-0.20), which the
    # per-k fill used to count while dropping the k1 state at -0.50.
    at = lambda e: float(dos[int(np.argmin(np.abs(grid - e)))])
    assert at(-0.20) < 1.0e-6
    assert at(-0.50) > 1.0


# ---------------------------------------------------------------------------
# The native twin (compute_kpoint_band_edges) measures the same thing
# ---------------------------------------------------------------------------

_ANG = 1.8897259886


def _compressed_h2(dims=(1.4, 1.8, 3.0)):
    from vibeqc._vibeqc_core import Atom, PeriodicSystem

    lx, ly, lz = (d * _ANG for d in dims)
    return PeriodicSystem(
        3,
        np.diag([lx, ly, lz]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.5 * lx, 0.37 * ly, 0.29 * lz])],
        0,
        1,
    )


def _native_and_python(system, mesh):
    from vibeqc._vibeqc_core import monkhorst_pack
    from vibeqc._vibeqc_core import semiempirical as se

    kmesh = monkhorst_pack(system, list(mesh))
    result = se.run_dftb0_kpoints(
        system, se.SemiempiricalParameters.dftb0_default(), kmesh, 20.0
    )
    python = kpoint_band_edges(
        [_channel(
            [np.asarray(e) for e in result.eps_per_k],
            [np.asarray(o) for o in result.occupations_per_k],
        )],
        weights=list(kmesh.weights),
    )
    return result, python


@pytest.mark.parametrize(
    "dims, mesh",
    [
        ((1.4, 1.8, 3.0), (2, 2, 2)),  # integer crossing, counts {0, 2}
        ((1.4, 1.8, 3.0), (4, 1, 1)),  # integer crossing, counts {0, 1, 2}
        ((2.0, 2.4, 2.8), (3, 3, 3)),  # gapped
        ((2.0, 2.4, 2.8), (3, 2, 2)),  # fractional Fermi group
    ],
)
def test_native_band_edges_equal_the_python_classifier(dims, mesh):
    result, python = _native_and_python(_compressed_h2(dims), mesh)
    for name in (
        "valence_band_max",
        "conduction_band_min",
        "indirect_gap",
        "direct_gap",
        "gap_above_fermi_manifold",
    ):
        assert float(getattr(result, name)) == getattr(python, name), name
    for name in ("valence_band_max_k", "conduction_band_min_k", "direct_gap_k"):
        assert int(getattr(result, name)) == getattr(python, name), name
    assert bool(result.is_metallic) is python.is_metallic
    for name in ("valence_max_per_k", "conduction_min_per_k"):
        np.testing.assert_array_equal(
            np.asarray(list(getattr(result, name)), dtype=float),
            np.asarray(getattr(python, name), dtype=float),
            err_msg=name,
        )  # NaN where a k has no state of the class, on both sides


def test_native_k_route_reads_a_band_crossing_metal_as_metallic():
    """Before #176 the native record of this fill read ``is_metallic=False``
    with a positive gap: the state classes of an integer global fill never
    overlap."""
    result, python = _native_and_python(_compressed_h2(), (2, 2, 2))
    occ = [np.asarray(o) for o in result.occupations_per_k]
    assert not any(np.any((o > 1e-8) & (o < 2 - 1e-8)) for o in occ)
    assert len({int(np.count_nonzero(o > 1e-8)) for o in occ}) > 1
    assert python.fermi_level_crosses_band
    assert result.is_metallic is True
    assert float(result.indirect_gap) < 0.0
    # The pre-#176 reading of the same record: positive, "insulating".
    _, _, state_class_gap = _state_class_edges(
        [np.asarray(e) for e in result.eps_per_k], occ
    )
    assert state_class_gap > 0.0
    assert float(result.gap_above_fermi_manifold) == pytest.approx(
        state_class_gap
    )


# ---------------------------------------------------------------------------
# Real SCF results
# ---------------------------------------------------------------------------


def test_run_periodic_job_out_system_and_accessor_agree(tmp_path):
    """A real multi-k SCF through the runner: the ``.out`` band block, the
    ``.system`` ``[run]`` band-edge fields and ``frontier_bands`` read one
    record. Before #176 the ab-initio ``.system`` carried no band edges."""
    import tomllib

    box = 18.0
    c = box / 2
    system = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "h2_bipole_k211"
    result = vq.run_periodic_job(
        system, basis, method="RHF", jk_method="bipole", kpoints=(2, 1, 1),
        max_iter=60, output=stem, output_qvf=False, write_density=False,
        citations=False, progress=False,
    )
    frontier = vq.frontier_bands(result)
    run = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))["run"]
    text = stem.with_suffix(".out").read_text("utf-8")

    assert not frontier.is_metallic
    assert run["is_metallic"] is False
    assert run["indirect_gap"] == pytest.approx(frontier.gap, abs=1e-12)
    assert run["valence_band_max"] == pytest.approx(frontier.homo, abs=1e-12)
    assert "Band extrema (multi-k)" in text
    assert f"{frontier.gap:.6f} Ha" in text
    # The gapped reading is the band-index reading.
    assert vq.frontier_bands(result, n_occ=1).gap == pytest.approx(frontier.gap)
    # BIPOLE RHF records the Fermi level of the fill it used (midgap here).
    assert frontier.homo < result.fermi_level < frontier.lumo


@pytest.mark.slow
def test_bipole_uhf_singlet_reproduces_rhf_on_a_band_crossing_metal():
    """Driver-level #176 gate on a real band-overlap metal.

    The compressed low-symmetry H2 cell at (2,2,2) converges to a state
    whose fill crosses a band (occupied counts {0, 2} per k). Before #176
    BIPOLE RHF sliced ``C[:, :n_occ]`` at every k and BIPOLE UHF filled each
    spin per k, so neither could represent it. Now both take the shared
    global fill, UHF(M=1) reproduces RHF, the density is built from the
    occupations, and the classifier reads the metal (measured 2026-09-27:
    E = -0.9539543112 Ha RHF, -0.9539543115 Ha UHF, band overlap
    -1.09944 Ha; the multi-k GDF route gives -0.9539541 Ha on the same cell).
    """
    system = vq.PeriodicSystem(
        3,
        np.diag([1.4 * _ANG, 1.8 * _ANG, 3.0 * _ANG]),
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.7 * _ANG, 0.37 * 1.8 * _ANG, 0.29 * 3.0 * _ANG]),
        ],
        0,
        1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, [2, 2, 2])
    rhf = vq.run_pbc_bipole_rhf(system, basis, kmesh, progress=False)
    uhf = vq.run_pbc_bipole_uhf(system, basis, kmesh, progress=False)

    assert rhf.converged and uhf.converged
    counts = {int(np.count_nonzero(np.asarray(o) > 1e-8)) for o in rhf.occupations}
    assert len(counts) > 1  # the fill crosses a band
    assert uhf.energy == pytest.approx(rhf.energy, abs=1e-8)
    for result in (rhf, uhf):
        edges = result_band_edges(result)
        assert edges.fermi_level_crosses_band
        assert edges.is_metallic
        assert edges.indirect_gap < 0.0
    assert uhf.fermi_level_alpha == pytest.approx(rhf.fermi_level, abs=1e-6)
