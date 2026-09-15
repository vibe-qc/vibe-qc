"""Measured band edges + gaps on the semiempirical k-route result surface.

Issue #426: the k-route computed the indirect gap internally but stored only
``fermi_level`` -- judges were forced onto torus-gap certificates and 159
dual1d pairs were lost for want of a measured gap.  These gates pin the
result-surface contract:

* ``indirect_gap``, ``direct_gap``, ``valence_band_max``,
  ``conduction_band_min``, their k indices, and the per-k band edges are
  native result-struct fields on all three k-route results
  (DFTB0 / SCC-DFTB / GFN2).
* The classification derives from the occupations actually used
  (``occupations_per_k``), with the same masks as the ab-initio
  ``_band_summary``: a state is valence when ``n > 1e-8`` and conduction
  when ``n < 2 - 1e-8``; a fractionally occupied state (degenerate
  zero-temperature Fermi group or finite-T smearing) belongs to BOTH
  classes, so any fractional Fermi group forces ``indirect_gap <= 0`` --
  converged metals measure an explicit 0.0-or-negative gap, never an
  absent one.
* NaN appears only where an edge does not exist in the model space (no
  occupied state / completely filled valence-minimal basis) or where the
  SCF did not converge (nothing was measured).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc._vibeqc_core import (
    Atom,
    PeriodicSystem,
    bloch_kmesh_from_lists,
    monkhorst_pack,
)
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.molecule import ANGSTROM_TO_BOHR
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params


_CHAIN_LENGTH = 4.1
_CUTOFF = 12.0
_SI_CUTOFF = 40.0
_SI_REPEATS = 3

# The documented occupation classification threshold (issue #426); matches
# the ab-initio _band_summary masks in periodic_runner.py.
_OCC_TOL = 1.0e-8


def _hli_chain(charge: int = 0) -> PeriodicSystem:
    return PeriodicSystem(
        1,
        np.diag([_CHAIN_LENGTH, 30.0, 30.0]),
        [
            Atom(1, [0.17, 0.31, 0.0]),
            Atom(3, [1.39, -0.22, 0.0]),
        ],
        charge,
        1,
    )


def _diamond_si() -> PeriodicSystem:
    a = 5.43 * ANGSTROM_TO_BOHR
    primitive_lattice = np.array(
        [
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        ]
    )
    return PeriodicSystem(
        3,
        primitive_lattice,
        [Atom(14, np.zeros(3)), Atom(14, np.full(3, a / 4.0))],
        0,
        1,
    )


def _filled_argon() -> PeriodicSystem:
    return PeriodicSystem(
        3,
        np.diag([10.0, 10.0, 10.0]),
        [Atom(18, [0.0, 0.0, 0.0])],
        0,
        1,
    )


def _graphene_2c() -> PeriodicSystem:
    a = 4.65
    return PeriodicSystem(
        2,
        np.array(
            [[a, 0.0, 0.0], [a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 20.0]]
        ),
        [
            Atom(6, [0.0, 0.0, 0.0]),
            Atom(6, [a / 2.0, a * np.sqrt(3.0) / 6.0, 0.0]),
        ],
        0,
        1,
    )


def _hbn_2c() -> PeriodicSystem:
    """Gapped 2-atom h-BN monolayer: a well-defined T=0 Aufbau state.

    Graphene's Gamma frontier is a symmetry-degenerate doublet holding two
    electrons, so the exact zero-temperature Aufbau problem it poses has no
    solution and its SCC cannot converge at T=0 (measured 2026-09-06: the
    two frontier levels split by 6e-5 Ha, occupations 1.16/0.84 at T=1e-4).
    The isoelectronic h-BN cell breaks that degeneracy with a 0.149 Ha gap.
    """
    a = 4.732
    return PeriodicSystem(
        2,
        np.array(
            [[a, 0.0, 0.0], [a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 20.0]]
        ),
        [
            Atom(5, [0.0, 0.0, 0.0]),
            Atom(7, [a / 2.0, a * np.sqrt(3.0) / 6.0, 0.0]),
        ],
        0,
        1,
    )


def _derive_edges(eps_per_k, occupations_per_k):
    """Pure-python re-derivation of the documented #426 edge convention."""
    nan = float("nan")
    vbm, cbm = -np.inf, np.inf
    vbm_k = cbm_k = direct_k = -1
    direct = np.inf
    vmax_per_k, cmin_per_k = [], []
    for k_idx, (eps, occ) in enumerate(zip(eps_per_k, occupations_per_k)):
        eps = np.asarray(eps, dtype=float)
        occ = np.asarray(occ, dtype=float)
        occupied = eps[occ > _OCC_TOL]
        conduction = eps[occ < 2.0 - _OCC_TOL]
        vmax = float(np.max(occupied)) if occupied.size else None
        cmin = float(np.min(conduction)) if conduction.size else None
        vmax_per_k.append(nan if vmax is None else vmax)
        cmin_per_k.append(nan if cmin is None else cmin)
        if vmax is not None and vmax > vbm:
            vbm, vbm_k = vmax, k_idx
        if cmin is not None and cmin < cbm:
            cbm, cbm_k = cmin, k_idx
        if vmax is not None and cmin is not None and cmin - vmax < direct:
            direct, direct_k = cmin - vmax, k_idx
    return {
        "valence_band_max": vbm if vbm_k >= 0 else nan,
        "conduction_band_min": cbm if cbm_k >= 0 else nan,
        "indirect_gap": cbm - vbm if (vbm_k >= 0 and cbm_k >= 0) else nan,
        "direct_gap": direct if direct_k >= 0 else nan,
        "valence_band_max_k": vbm_k,
        "conduction_band_min_k": cbm_k,
        "direct_gap_k": direct_k,
        "valence_max_per_k": vmax_per_k,
        "conduction_min_per_k": cmin_per_k,
    }


def _assert_edges_match_rederivation(result, *, abs_tol: float = 1.0e-12):
    derived = _derive_edges(result.eps_per_k, result.occupations_per_k)
    for name in ("valence_band_max", "conduction_band_min", "indirect_gap", "direct_gap"):
        native = float(getattr(result, name))
        expected = derived[name]
        if math.isnan(expected):
            assert math.isnan(native), f"{name}: expected NaN, got {native!r}"
        else:
            assert native == pytest.approx(expected, abs=abs_tol), name
    for name in ("valence_band_max_k", "conduction_band_min_k", "direct_gap_k"):
        assert int(getattr(result, name)) == derived[name], name
    for name in ("valence_max_per_k", "conduction_min_per_k"):
        native_vector = list(getattr(result, name))
        expected_vector = derived[name]
        assert len(native_vector) == len(expected_vector) == len(result.eps_per_k)
        for native, expected in zip(native_vector, expected_vector):
            if math.isnan(expected):
                assert math.isnan(native), name
            else:
                assert native == pytest.approx(expected, abs=abs_tol), name
    # min_k(per-k gap) can never undercut the global CBM - VBM.
    if not math.isnan(derived["indirect_gap"]) and not math.isnan(derived["direct_gap"]):
        assert float(result.direct_gap) >= float(result.indirect_gap) - abs_tol
    return derived


@pytest.fixture(scope="module")
def parameters():
    return _se.SemiempiricalParameters.dftb0_default()


def test_dftb0_gapped_chain_reports_positive_indirect_gap(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF)

    # Integer zero-temperature Aufbau: [2.0, 0.0] at every k.
    for occupation in result.occupations_per_k:
        np.testing.assert_array_equal(occupation, [2.0, 0.0])

    _assert_edges_match_rederivation(result)

    # eps_per_k arithmetic for the integer-occupation case.
    vbm = max(float(energies[0]) for energies in result.eps_per_k)
    cbm = min(float(energies[1]) for energies in result.eps_per_k)
    assert float(result.valence_band_max) == vbm
    assert float(result.conduction_band_min) == cbm
    assert float(result.indirect_gap) == pytest.approx(cbm - vbm, abs=1.0e-15)
    # The known HLi/DFTB0 3x1x1 gap, measured on this fixture at the #426
    # fix (~2.08 eV): a robustly positive insulator value.
    assert float(result.indirect_gap) == pytest.approx(0.076425987187, abs=1.0e-9)


def test_dftb0_metallic_si_measures_nonpositive_gap(parameters):
    """The #317 diamond-Si 3x3x3 family: a fractional global-Aufbau Fermi
    group must measure an explicit <= 0 indirect gap, never an absent one."""
    system = _diamond_si()
    kmesh = monkhorst_pack(system, (_SI_REPEATS, _SI_REPEATS, _SI_REPEATS))
    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _SI_CUTOFF)

    occupations = np.concatenate(
        [np.asarray(occupation) for occupation in result.occupations_per_k]
    )
    fractional = (occupations > _OCC_TOL) & (occupations < 2.0 - _OCC_TOL)
    assert np.count_nonzero(fractional) == 8  # the shared 0.5 Fermi group

    _assert_edges_match_rederivation(result)

    gap = float(result.indirect_gap)
    assert math.isfinite(gap)
    assert gap <= 0.0
    # The Fermi group is roundoff-degenerate, so the measured overlap is
    # bounded by the kernel's 256*eps energy tolerance, not by -5e-5.
    assert gap > -1.0e-10


def _pooled_aufbau_gap(result) -> float:
    """The Gamma route's own convention, recomputed as an external judge
    would: sort every state, fill the lowest ``n_occ * n_k``, and report
    ``eps[n_fill] - eps[n_fill - 1]``."""
    flat = sorted(
        float(np.asarray(eps)[band])
        for eps in result.eps_per_k
        for band in range(len(np.asarray(eps)))
    )
    n_fill = result.n_occ * result.n_kpoints
    return flat[n_fill] - flat[n_fill - 1]


def _occupancy_partition_gap(result) -> float:
    """The occupancy-partition quantity: ``min(eps | n <= t) -
    max(eps | n > t)``. This is what the pre-#426 wave payloads reported
    as "gap" -- the sec8-r4 F1 defect."""
    eps = np.concatenate([np.asarray(e) for e in result.eps_per_k])
    occ = np.concatenate([np.asarray(o) for o in result.occupations_per_k])
    return float(eps[occ <= _OCC_TOL].min() - eps[occ > _OCC_TOL].max())


def test_degenerate_fermi_edge_gap_matches_pooled_aufbau(parameters):
    """Pin the meaning of the gap at a FRACTIONALLY OCCUPIED DEGENERATE
    edge -- the case the 2026-08-27 sec8-r4 judge measured as defective in
    the pre-#426 wave records (finding F1).

    Diamond-Si 3x3x3's global Fermi level is pinned inside a manifold of 8
    states sharing occupation 0.5. The physically correct gap there is
    exactly 0, and the manifold is degenerate to machine precision. Three
    quantities are distinguished:

    * ``indirect_gap`` (shipped): the occupation convention -- every
      manifold member is both valence and conduction, so the value is the
      negative manifold spread, i.e. 0 to machine precision. CORRECT.
    * pooled aufbau (the Gamma route's convention): also 0 to machine
      precision. The judge's recommended semantics; this test pins that
      the shipped field MATCHES it, so k-route and Gamma-route duality
      rungs compare like with like.
    * occupancy partition (the pre-#426 payload formula): steps over the
      whole partially filled manifold and returns a strictly positive
      number -- here ~5.7e-5 Ha, inside the judge's measured 8.06e-5 Ha
      median / 1.98e-2 Ha maximum error band. Shipped only under its own
      name, ``gap_above_fermi_manifold``, never as ``gap``.
    """
    system = _diamond_si()
    kmesh = monkhorst_pack(system, (_SI_REPEATS, _SI_REPEATS, _SI_REPEATS))
    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _SI_CUTOFF)

    occupations = np.concatenate(
        [np.asarray(occupation) for occupation in result.occupations_per_k]
    )
    fractional = (occupations > _OCC_TOL) & (occupations < 2.0 - _OCC_TOL)
    assert np.count_nonzero(fractional) == 8, "expected the shared Fermi group"

    # The manifold is degenerate to machine precision (judge: <= 8.7e-15 Ha
    # across 1,668 rows), so "the gap is exactly 0" is not a borderline call.
    manifold = np.concatenate(
        [np.asarray(e) for e in result.eps_per_k]
    )[fractional]
    assert float(manifold.max() - manifold.min()) < 1.0e-14

    native_gap = float(result.indirect_gap)
    pooled = _pooled_aufbau_gap(result)
    partition = _occupancy_partition_gap(result)

    # The shipped gap IS the pooled-aufbau gap, to machine precision.
    assert native_gap == pytest.approx(pooled, abs=1.0e-14)
    assert abs(native_gap) < 1.0e-14
    # ... and is never the spuriously positive partition quantity. Under a
    # global aufbau fill the partition formula is >= 0 by construction --
    # the structural reason the judge's negative-gap screen was vacuous on
    # every mesh row it examined.
    assert partition > 1.0e-5
    assert abs(native_gap - partition) > 1.0e-5
    # That quantity is still available -- under its own name only.
    assert float(result.gap_above_fermi_manifold) == pytest.approx(
        partition, abs=1.0e-15
    )

    # The gapless case is STRUCTURALLY visible, not inferred from a float.
    assert result.is_metallic is True

    # The edges are the manifold itself: both classes see the same states.
    assert float(result.valence_band_max) == pytest.approx(
        float(result.conduction_band_min), abs=1.0e-14
    )


def test_integer_filled_row_has_no_convention_regression_surface(parameters):
    """On an integer-filled (genuinely gapped) row every convention agrees
    -- the judge measured 646/646 such rows in agreement, so adopting the
    correct convention has no regression surface."""
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF)

    occupations = np.concatenate(
        [np.asarray(occupation) for occupation in result.occupations_per_k]
    )
    assert not np.any((occupations > _OCC_TOL) & (occupations < 2.0 - _OCC_TOL))

    native_gap = float(result.indirect_gap)
    assert native_gap == pytest.approx(_pooled_aufbau_gap(result), abs=1.0e-15)
    assert native_gap == pytest.approx(
        _occupancy_partition_gap(result), abs=1.0e-15
    )
    assert float(result.gap_above_fermi_manifold) == pytest.approx(
        native_gap, abs=1.0e-15
    )
    assert result.is_metallic is False


def test_dftb0_bandpath_band_overlap_measures_negative_gap(parameters):
    """Independent per-point filling on a band-overlapping metal: the
    occupation-derived indirect gap is the (negative) HOMO/LUMO overlap."""
    system = _diamond_si()
    kpath = monkhorst_pack(
        system, (_SI_REPEATS, _SI_REPEATS, _SI_REPEATS)
    ).kpoints
    result = _se.run_dftb0_bandpath(system, parameters, kpath, _SI_CUTOFF)

    for occupation in result.occupations_per_k:
        np.testing.assert_array_equal(
            occupation, [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        )

    _assert_edges_match_rederivation(result)

    homo = max(float(energies[result.n_occ - 1]) for energies in result.eps_per_k)
    lumo = min(float(energies[result.n_occ]) for energies in result.eps_per_k)
    assert float(result.indirect_gap) == pytest.approx(lumo - homo, abs=1.0e-15)
    assert float(result.indirect_gap) < -5.0e-5

    # ANTI-VACUITY (sec8-r4): a "negative gap" screen is a tautology
    # against the pooled-aufbau formula, which is >= 0 by construction
    # (adjacent members of one sorted list). That is why the pre-#426 wave
    # reported 0 negative gaps in 1,625 rows and the #317 screen could
    # never fail. The shipped field CAN be genuinely negative, so the
    # screen is able to fail and its passing now carries information.
    assert _pooled_aufbau_gap(result) >= 0.0
    assert float(result.indirect_gap) < 0.0
    assert result.is_metallic is True


def test_dftb0_smeared_edges_match_python_rederivation(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (4, 1, 1))
    options = _se.KPointOccupationOptions()
    options.smearing_temperature = 0.05
    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF, options)

    derived = _assert_edges_match_rederivation(result)
    assert math.isfinite(float(result.indirect_gap))
    assert math.isfinite(derived["indirect_gap"])


def test_dftb0_zero_weight_kpoints_participate_in_edges(parameters):
    """Zero-weight points are populated from the same global step function
    and are part of the measured spectrum (band-structure spelling)."""
    system = _hli_chain()
    points = monkhorst_pack(system, (3, 1, 1)).kpoints
    kmesh = bloch_kmesh_from_lists(points, [0.0, 1.0, 0.0])
    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF)

    derived = _assert_edges_match_rederivation(result)
    assert len(list(result.valence_max_per_k)) == 3
    assert not math.isnan(derived["valence_max_per_k"][0])


def test_dftb0_filled_argon_has_no_conduction_edge(parameters):
    """A completely filled valence-minimal basis has no conduction state:
    the edge does not exist in the model space and reports NaN, with the
    valence edge still measured."""
    system = _filled_argon()
    kmesh = monkhorst_pack(system, (2, 2, 2))
    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF)

    for occupation in result.occupations_per_k:
        np.testing.assert_array_equal(occupation, [2.0, 2.0, 2.0, 2.0])

    _assert_edges_match_rederivation(result)
    assert math.isnan(float(result.conduction_band_min))
    assert math.isnan(float(result.indirect_gap))
    assert math.isnan(float(result.direct_gap))
    assert int(result.conduction_band_min_k) == -1
    assert int(result.direct_gap_k) == -1
    assert all(math.isnan(value) for value in result.conduction_min_per_k)
    vbm = float(result.valence_band_max)
    assert vbm == max(float(np.max(np.asarray(e))) for e in result.eps_per_k)
    assert int(result.valence_band_max_k) >= 0


def test_dftb0_empty_valence_has_no_valence_edge(parameters):
    system = _hli_chain(charge=2)
    kmesh = monkhorst_pack(system, (3, 1, 1))
    result = _se.run_dftb0_kpoints(system, parameters, kmesh, _CUTOFF)

    _assert_edges_match_rederivation(result)
    assert math.isnan(float(result.valence_band_max))
    assert math.isnan(float(result.indirect_gap))
    assert int(result.valence_band_max_k) == -1
    assert math.isfinite(float(result.conduction_band_min))


def test_scc_dftb_gapped_chain_edges_consistent(parameters):
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    options = _se.SCCOptions()
    options.max_iter = 300
    options.conv_tol_charge = 1.0e-9
    result = _se.run_scc_dftb_kpoints(system, parameters, kmesh, options, _CUTOFF)

    assert result.converged
    _assert_edges_match_rederivation(result)
    # The threshold was calibrated on the Klopman-Ohno surface.  Since D1
    # (#425) the periodic gamma is the Ewald-split Elstner form, under which
    # this chain is more ionic and its gap is 0.0478 Ha rather than >0.1.
    # The subject of the test is that a gapped chain reports consistent
    # edges, so the assertion keeps a clearly positive gap -- an order of
    # magnitude above the roundoff-degeneracy scale the edge machinery uses
    # -- rather than the old surface's magnitude.
    assert float(result.indirect_gap) > 0.01


def test_scc_dftb_unconverged_record_measures_no_gap(parameters):
    """An unconverged diagnostics record (#342 opt-in) measured nothing:
    its gap fields are NaN, never a fake 0.0 that screens as a metal."""
    system = _hli_chain()
    kmesh = monkhorst_pack(system, (3, 1, 1))
    options = _se.SCCOptions()
    options.max_iter = 1
    options.conv_tol_charge = 1.0e-12
    result = _se.run_scc_dftb_kpoints(
        system,
        parameters,
        kmesh,
        options,
        _CUTOFF,
        _se.KPointOccupationOptions(),
        True,  # allow_unconverged: diagnostics record, flagged converged=False
    )

    assert not result.converged
    assert math.isnan(float(result.indirect_gap))
    assert math.isnan(float(result.direct_gap))
    assert math.isnan(float(result.valence_band_max))
    assert math.isnan(float(result.conduction_band_min))
    assert int(result.valence_band_max_k) == -1
    assert int(result.conduction_band_min_k) == -1
    assert int(result.direct_gap_k) == -1
    assert list(result.valence_max_per_k) == []
    assert list(result.conduction_min_per_k) == []


def test_gfn2_gamma_mesh_exposes_edges_and_occupations():
    """The GFN2 one-point-Gamma spelling carries occupations_per_k and the
    same measured edges as a re-derivation from its own spectrum.

    Run on gapped h-BN rather than graphene: this is the exact-T=0 spelling,
    and graphene's Gamma frontier is a partly filled symmetry-degenerate
    doublet with no Aufbau ground state (see _hbn_2c).  Graphene keeps the
    smeared sibling below, which is the path it has one.
    """
    system = _hbn_2c()
    params = load_gfn2_params()
    gamma_mesh = bloch_kmesh_from_lists([[0.0, 0.0, 0.0]], [1.0])
    options = _xtb.XTBSccOptions()
    options.electronic_temperature = 0.0
    options.max_iter = 400
    result = _xtb.run_gfn2_xtb_kpoints(system, params, gamma_mesh, options, 12.0)

    assert result.converged
    assert len(list(result.occupations_per_k)) == 1
    occ = np.asarray(result.occupations_per_k[0])
    assert occ.shape == (result.n_basis,)
    assert float(occ.sum()) == pytest.approx(2.0 * result.n_occ, abs=1.0e-10)

    _assert_edges_match_rederivation(result)
    assert math.isfinite(float(result.indirect_gap))
    # One k-point: the indirect and direct gaps coincide.
    assert float(result.direct_gap) == pytest.approx(
        float(result.indirect_gap), abs=1.0e-15
    )
    assert int(result.direct_gap_k) == 0


def test_gfn2_default_smearing_edges_follow_occupations():
    system = _graphene_2c()
    params = load_gfn2_params()
    gamma_mesh = bloch_kmesh_from_lists([[0.0, 0.0, 0.0]], [1.0])
    options = _xtb.XTBSccOptions()
    options.max_iter = 400
    result = _xtb.run_gfn2_xtb_kpoints(system, params, gamma_mesh, options, 12.0)

    assert result.converged
    _assert_edges_match_rederivation(result)
