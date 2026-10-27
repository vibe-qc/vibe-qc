"""Explicit spin-state contract of the neutral Γ-CCM producers (#519, RG-14).

``CCMSystem(..., spin_state=)`` resolves the torus spin sector before the
shell-aware dispatch: ``"legacy"`` reproduces each producer's historical
convention (the real-Gamma supercell pairs a doublet cell by parity, the
Bloch producer replicates the cell's counts at every k point), ``"per-cell"``
makes both execute the cell's counts replicated with spins aligned, and an
integer is the total torus multiplicity. The executed sector is recorded on
every result; a comparison needs one operator **and** one sector.

Rows: the resolution table on the Li doublet cube and an H2 triplet cell;
impossible requests refused at construction; the legacy mismatch reproduced
with its counts and refused as a comparison; the same per-cell sector on both
producers at ``(2,1,1)`` and ``(3,1,1)`` (Li) and on the multidimensional
``(2,2,1)`` / ``(2,2,2)`` tori (the H cube) to 1e-10 Ha/cell with the Bloch
projectors folded onto the torus; UKS PBE at ``(2,1,1)``; restricted drivers
refused on an open sector; READ of a foreign-sector density refused; the
record round trip and the runner manifest. RG-15: the occupation record on
both producers (integer or ensemble, per-k counts, frontier gap); the Li
quintet at ``(2,2,1)``, an ensemble on the Bloch side, refused by name before
any fold; the manifest carries the record.
"""
from __future__ import annotations

import json
import tomllib
import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicRHFOptions, PeriodicSystem
from vibeqc.periodic.ccm import (
    CCMSpinState,
    CCMSystem,
    OccupationRecord,
    assert_integer_occupations,
    assert_comparable_neutral_results,
    assert_compatible_spin_state,
    assert_matched_neutral_source,
    ccm_reference_from_bloch,
    resolve_ccm_spin_state,
    spin_state_of,
)
from vibeqc.periodic.ccm.direct import (
    run_ccm_rhf_direct,
    run_ccm_uhf_direct,
    run_ccm_uks_direct,
)
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_uhf_gdf, run_ccm_uks_gdf
from vibeqc.periodic.ccm.route import run_ccm_scf
from vibeqc.periodic.ccm.spin_state import bloch_executed_state

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

ENERGY_TOL_HA_PER_CELL = 1e-10
#: Both producers converge far below the default commutator bound, under
#: which two SCF paths of one sector may stop at densities 1e-6 apart.
TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9, max_iter=300)


def _li_doublet():
    return PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)


def _h_doublet():
    return PeriodicSystem(
        3, np.diag([5.0, 5.0, 5.0]), [Atom(1, [2.5, 2.5, 2.5])], 0, 2)


def _h2_cube(mult=1):
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, mult)


def _bloch_options():
    opts = PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-8
    opts.max_iter = 200
    return opts


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


# ---------------------------------------------------------------- resolution

@pytest.mark.parametrize("nrep, policy, expected", [
    ((1, 1, 1), "legacy", (2, 1, 2, "parity-paired")),
    ((1, 1, 1), "per-cell", (2, 1, 2, "uniform")),
    ((2, 1, 1), "legacy", (3, 3, 1, "parity-paired")),
    ((2, 1, 1), "per-cell", (4, 2, 3, "uniform")),
    ((2, 1, 1), 3, (4, 2, 3, "explicit-total")),
    ((3, 1, 1), "legacy", (5, 4, 2, "parity-paired")),
    ((3, 1, 1), "per-cell", (6, 3, 4, "uniform")),
    ((2, 2, 1), "legacy", (6, 6, 1, "parity-paired")),
    ((2, 2, 1), "per-cell", (8, 4, 5, "uniform")),
    ((2, 2, 1), 3, (7, 5, 3, "explicit-total")),
])
def test_doublet_cell_resolution_table(nrep, policy, expected):
    """The three policies on the Li doublet cube, every mesh parity."""
    state = resolve_ccm_spin_state(_li_doublet(), nrep, policy)
    n_a, n_b, mult, arrangement = expected
    assert (state.n_alpha, state.n_beta, state.multiplicity, state.arrangement) == (
        n_a, n_b, mult, arrangement)
    assert state.n_electrons == 3 * int(np.prod(nrep))
    # The cluster carries the same state and its supercell multiplicity.
    ccm = CCMSystem(_li_doublet(), nrep, "sto-3g", spin_state=policy)
    assert ccm.spin_state == state
    assert int(ccm.supercell.multiplicity) == mult
    assert state.uniform == (n_a % state.n_cells == 0 and n_b % state.n_cells == 0)


def test_legacy_rule_is_the_historical_supercell_rule():
    """``"legacy"`` reproduces the pinned conventions of ``test_ccm_uhf``."""
    li = _li_doublet()
    assert CCMSystem(li, (1, 1, 1), "sto-3g").supercell.multiplicity == 2
    assert CCMSystem(li, (2, 1, 1), "sto-3g").supercell.multiplicity == 1
    trip = _h2_cube(mult=3)
    assert CCMSystem(trip, (1, 1, 1), "sto-3g").supercell.multiplicity == 3
    assert CCMSystem(trip, (2, 1, 1), "sto-3g").supercell.multiplicity == 5
    # An explicit high spin is already uniform under legacy: per-cell agrees.
    legacy = resolve_ccm_spin_state(trip, (2, 1, 1), "legacy")
    per_cell = resolve_ccm_spin_state(trip, (2, 1, 1), "per-cell")
    assert (legacy.n_alpha, legacy.n_beta) == (per_cell.n_alpha, per_cell.n_beta) == (4, 0)
    # A singlet cell is one sector under every policy.
    for policy in ("legacy", "per-cell", 1):
        state = resolve_ccm_spin_state(_h2_cube(), (2, 2, 1), policy)
        assert state.closed_shell and (state.n_alpha, state.n_beta) == (4, 4)


@pytest.mark.parametrize("system, nrep, policy, match", [
    (_li_doublet(), (1, 1, 1), 3, "incompatible"),        # 3 electrons, triplet
    (_li_doublet(), (3, 1, 1), 3, "incompatible"),        # 9 electrons, triplet
    (_li_doublet(), (2, 1, 1), 9, "unpaired"),            # 6 electrons, 8 unpaired
    (_li_doublet(), (2, 1, 1), 0, ">= 1"),
    (_li_doublet(), (2, 1, 1), "antiferro", "spin_state must be"),
    (_li_doublet(), (2, 1, 1), True, "spin_state must be"),
    (_h2_cube(mult=4), (1, 1, 1), "legacy", "incompatible"),
    (_h2_cube(mult=2), (1, 1, 1), "per-cell", "incompatible"),
])
def test_impossible_requests_fail_at_construction(system, nrep, policy, match):
    with pytest.raises(ValueError, match=match):
        CCMSystem(system, nrep, "sto-3g", spin_state=policy)


def test_bloch_refuses_a_sector_that_is_not_uniform_per_cell():
    """The Bloch representation holds N_c copies of one pair only."""
    state = resolve_ccm_spin_state(_li_doublet(), (2, 2, 1), 3)  # 7/5 on 4 cells
    assert not state.uniform and state.per_cell is None
    with pytest.raises(ValueError, match="not uniform per cell"):
        bloch_executed_state(state, who="run_ccm_uhf_gdf")
    # Under legacy the Bloch producer executes the unit-cell convention.
    legacy = resolve_ccm_spin_state(_li_doublet(), (2, 1, 1), "legacy")
    executed = bloch_executed_state(legacy, who="run_ccm_uhf_gdf")
    assert (executed.n_alpha, executed.n_beta) == (4, 2)
    assert executed.producer == "neutral-bloch" and executed.arrangement == "uniform"
    ccm = CCMSystem(_li_doublet(), (2, 2, 1), "sto-3g", spin_state=3)
    with pytest.raises(ValueError, match="not uniform per cell"):
        _quiet(run_ccm_uhf_gdf, ccm, options=_bloch_options())


def test_record_round_trip_and_comparison_fields():
    state = resolve_ccm_spin_state(_li_doublet(), (3, 1, 1), "per-cell")
    payload = json.loads(json.dumps(state.as_dict()))
    assert payload["schema"] == "vibeqc.ccm-spin-state.v1"
    assert payload["per_cell"] == [2, 1] and payload["multiplicity"] == 4
    back = CCMSpinState.from_dict(payload)
    assert back == state
    assert spin_state_of({"spin_state": payload}.__class__) is None
    # A record on a convention object is found through either arm's name.
    class _Conv:
        spin_state = payload

    class _Result:
        real_gamma = _Conv()

    assert spin_state_of(_Result()) == state
    with pytest.raises(ValueError, match="no spin-state record"):
        assert_compatible_spin_state([state, object()])


# ------------------------------------------------------- legacy mismatch

@pytest.fixture(scope="module")
def li_211_legacy():
    """The demonstrated deficiency, reproduced with its counts (#519 RG-8b)."""
    ccm = CCMSystem(_li_doublet(), (2, 1, 1), "sto-3g")
    rg = _quiet(run_ccm_uhf_direct, ccm, **TIGHT)
    nb = _quiet(run_ccm_uhf_gdf, ccm, options=_bloch_options())
    return ccm, rg, nb


def test_legacy_mismatch_is_two_sectors_of_one_operator(li_211_legacy):
    ccm, rg, nb = li_211_legacy
    assert rg.converged and nb.converged
    # Real-Gamma: the supercell singlet; Bloch: the cell's doublet twice.
    assert (rg.n_alpha, rg.n_beta) == (3, 3)
    assert rg.spin_state.arrangement == "parity-paired" and rg.spin_state.closed_shell
    assert (nb.spin_state.n_alpha, nb.spin_state.n_beta) == (4, 2)
    assert nb.spin_state.producer == "neutral-bloch"
    S = rg.overlap
    assert np.trace(rg.density_alpha @ S) == pytest.approx(3.0, abs=1e-8)
    assert np.trace(rg.density_beta @ S) == pytest.approx(3.0, abs=1e-8)
    assert abs(rg.s_squared) < 1e-6
    # One operator, two states: millihartrees apart, refused by sector.
    assert_matched_neutral_source([rg, nb])
    assert abs(rg.energy / ccm.n_cells - nb.energy) > 1e-3
    with pytest.raises(ValueError, match="different spin sectors"):
        assert_compatible_spin_state([rg, nb])
    with pytest.raises(ValueError, match="different spin sectors"):
        assert_comparable_neutral_results([rg, nb])


def test_restricted_drivers_refuse_an_open_sector():
    ccm = CCMSystem(_li_doublet(), (2, 1, 1), "sto-3g", spin_state="per-cell")
    with pytest.raises(ValueError, match="open shell"):
        _quiet(run_ccm_rhf_direct, ccm, **TIGHT)
    with pytest.raises(ValueError, match="open shell"):
        _quiet(run_ccm_rhf_gdf, ccm, options=_bloch_options())
    # The route dispatches on the resolved sector, not on the unit cell.
    trip = CCMSystem(_h2_cube(), (1, 1, 1), "sto-3g", spin_state=3)
    res = _quiet(run_ccm_scf, trip, route="real-gamma", **TIGHT)
    assert (res.n_alpha, res.n_beta) == (2, 0)
    assert res.spin_state.arrangement == "explicit-total"


# ------------------------------------------------- same sector, both producers

def _assert_same_sector_identity(ccm, rg, nb, *, fold=True):
    assert rg.converged and nb.converged
    identity, state = assert_comparable_neutral_results([rg, nb])
    assert identity and state.uniform
    # Both sides are determinants with one frontier (RG-15).
    occ_rg, occ_nb = assert_integer_occupations([rg, nb])
    assert occ_rg.integer and occ_nb.integer and occ_nb.n_blocks == ccm.n_cells
    assert occ_nb.frontier_gap == pytest.approx(occ_rg.frontier_gap, abs=1e-6)
    assert all(c == state.per_cell for c in
               [tuple(int(round(x)) for x in c) for c in occ_nb.counts_per_k])
    assert (rg.spin_state.n_alpha, rg.spin_state.n_beta) == (state.n_alpha, state.n_beta)
    if hasattr(rg, "n_alpha"):  # the UHF result; the KS result carries none
        assert (rg.n_alpha, rg.n_beta) == (state.n_alpha, state.n_beta)
    assert abs(rg.energy / ccm.n_cells - nb.energy) < ENERGY_TOL_HA_PER_CELL
    S = rg.overlap
    assert np.trace(rg.density_alpha @ S) == pytest.approx(state.n_alpha, abs=1e-8)
    assert np.trace(rg.density_beta @ S) == pytest.approx(state.n_beta, abs=1e-8)
    if not fold:
        return
    # The Bloch projectors folded onto the torus against the real-Gamma
    # projectors (gauge-free), and both stationary in their per-spin Focks.
    ref = _quiet(ccm_reference_from_bloch, ccm, nb, build_cderi=False)
    assert (ref.n_alpha, ref.n_beta) == (state.n_alpha, state.n_beta)
    np.testing.assert_allclose(ref.density_alpha, rg.density_alpha, atol=1e-6)
    np.testing.assert_allclose(ref.density_beta, rg.density_beta, atol=1e-6)
    assert ref.s_squared == pytest.approx(rg.s_squared, abs=1e-6)


@pytest.mark.parametrize("nrep", [(2, 1, 1), (3, 1, 1)])
def test_per_cell_doublet_on_even_and_odd_meshes(nrep):
    """Li doublet replicated with spins aligned: one sector, one energy."""
    ccm = CCMSystem(_li_doublet(), nrep, "sto-3g", spin_state="per-cell")
    rg = _quiet(run_ccm_uhf_direct, ccm, **TIGHT)
    nb = _quiet(run_ccm_uhf_gdf, ccm, options=_bloch_options())
    _assert_same_sector_identity(ccm, rg, nb)
    assert rg.spin_state.arrangement == nb.spin_state.arrangement == "uniform"
    # <S^2> at the aligned sector's S(S+1), up to the small contamination
    # of the open-shell determinant (1e-5 at (3,1,1)).
    assert rg.s_squared == pytest.approx(
        0.5 * nrep[0] * (0.5 * nrep[0] + 1.0), abs=1e-3)
    # The explicit total multiplicity of the same sector is the same run.
    explicit = CCMSystem(_li_doublet(), nrep, "sto-3g", spin_state=nrep[0] + 1)
    rg2 = _quiet(run_ccm_uhf_direct, explicit, **TIGHT)
    assert abs(rg2.energy - rg.energy) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL
    assert_compatible_spin_state([rg, rg2, nb])


@pytest.mark.parametrize("nrep", [(2, 2, 1), (2, 2, 2)])
def test_per_cell_doublet_on_multidimensional_meshes(nrep):
    """The H cube doublet on two- and three-dimensional tori."""
    ccm = CCMSystem(_h_doublet(), nrep, "sto-3g", spin_state="per-cell")
    rg = _quiet(run_ccm_uhf_direct, ccm, **TIGHT)
    nb = _quiet(run_ccm_uhf_gdf, ccm, options=_bloch_options())
    assert (rg.n_alpha, rg.n_beta) == (ccm.n_cells, 0)
    _assert_same_sector_identity(ccm, rg, nb)


def test_li_doublet_on_a_two_dimensional_mesh_has_a_degenerate_frontier():
    """Li (2,2,1) per cell: the Bloch alpha frontier is degenerate at the
    zone-boundary points and the multi-k driver occupies it fractionally,
    an ensemble the real-Gamma determinant cannot represent. The counts
    match, the fold refuses the occupation pattern: a same-sector
    comparison also needs integer occupations (recorded, not relaxed)."""
    ccm = CCMSystem(_li_doublet(), (2, 2, 1), "sto-3g", spin_state="per-cell")
    rg = _quiet(run_ccm_uhf_direct, ccm, **TIGHT)
    nb = _quiet(run_ccm_uhf_gdf, ccm, options=_bloch_options())
    # One operator, one sector ...
    assert_matched_neutral_source([rg, nb])
    assert_compatible_spin_state([rg, nb])
    # ... but an ensemble on the Bloch side: the record says so, the
    # comparison is refused before any fold (RG-15), and the fold agrees.
    occ = np.concatenate([np.asarray(o) for o in nb.raw.occupations_alpha])
    assert np.any((occ > 1e-8) & (occ < 1 - 1e-8))
    assert not nb.occupation.integer
    assert nb.occupation.max_fractional_deviation == pytest.approx(0.5, abs=1e-8)
    assert nb.occupation.frontier_gap is None
    assert rg.occupation.integer and rg.occupation.frontier_gap > 0.0
    with pytest.raises(ValueError, match="ensemble state"):
        assert_comparable_neutral_results([rg, nb])
    with pytest.raises(ValueError, match="non-integer occupations"):
        _quiet(ccm_reference_from_bloch, ccm, nb, build_cderi=False)
    payload = json.loads(json.dumps(nb.occupation.as_dict()))
    assert payload["schema"] == "vibeqc.ccm-occupation.v1"
    assert OccupationRecord.from_dict(payload) == nb.occupation
    with pytest.raises(ValueError, match="no occupation record"):
        assert_integer_occupations([rg, object()])


def test_per_cell_doublet_uks_pbe():
    ccm = CCMSystem(_li_doublet(), (2, 1, 1), "sto-3g", spin_state="per-cell")
    rg = _quiet(run_ccm_uks_direct, ccm, "pbe", **TIGHT)
    from vibeqc import PeriodicKSOptions

    opts = PeriodicKSOptions()
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-8
    opts.max_iter = 200
    nb = _quiet(run_ccm_uks_gdf, ccm, "pbe", options=opts)
    _assert_same_sector_identity(ccm, rg, nb, fold=False)
    assert rg.e_xc is not None and rg.fock_beta is not None


def test_read_of_a_foreign_sector_density_is_refused(li_211_legacy):
    """A READ seed carries its sector: the paired (3, 3) densities cannot
    start a (4, 2) run."""
    _, rg, _ = li_211_legacy
    ccm = CCMSystem(_li_doublet(), (2, 1, 1), "sto-3g", spin_state="per-cell")
    with pytest.raises(ValueError, match="electron count"):
        _quiet(run_ccm_uhf_direct, ccm, initial_guess="READ",
               initial_density=(rg.density_alpha, rg.density_beta), **TIGHT)


# ------------------------------------------------------------------ runner

def _run_fields(path):
    return tomllib.loads(path.with_suffix(".system").read_text(encoding="utf-8"))["run"]


def _job(tmp_path, name, system, **kw):
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.periodic_runner import run_periodic_job

    kw.setdefault("method", "aiccm")
    kw.setdefault("initial_guess", "HCORE")
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    return _quiet(run_periodic_job, system, basis, progress=False,
                  output=str(tmp_path / name), **kw)


def test_runner_carries_the_sector_on_both_variants(tmp_path):
    energies = {}
    for variant in ("real-gamma", "neutral-bloch"):
        res = _job(tmp_path, variant, _li_doublet(), variant=variant,
                   aiccm_lattice_extension=(2, 1, 1), aiccm_spin_state="per-cell")
        run = _run_fields(tmp_path / variant)
        assert run["aiccm_spin_policy"] == "per-cell"
        state = CCMSpinState.from_dict(run["aiccm_spin_state"])
        assert (state.n_alpha, state.n_beta, state.n_cells) == (4, 2, 2)
        assert run["aiccm_spin_state"]["uniform"] is True
        assert run["aiccm_spin_state"]["per_cell"] == [2, 1]
        occupation = OccupationRecord.from_dict(run["aiccm_occupation"])
        assert occupation.integer and occupation.frontier_gap > 0.0
        assert occupation.producer == variant
        energies[variant] = float(res.energy)
        header = (tmp_path / variant).with_suffix(".out").read_text()
        assert "spin state          = 'per-cell'" in header
    assert abs(energies["real-gamma"] - energies["neutral-bloch"]) < 1e-8
    # Legacy is the default and is recorded as such, with the paired counts.
    _job(tmp_path, "legacy", _li_doublet(), variant="real-gamma",
         aiccm_lattice_extension=(2, 1, 1))
    run = _run_fields(tmp_path / "legacy")
    assert run["aiccm_spin_policy"] == "legacy"
    assert run["aiccm_spin_state"]["arrangement"] == "parity-paired"
    assert (run["aiccm_spin_state"]["n_alpha"], run["aiccm_spin_state"]["n_beta"]) == (3, 3)
    assert run["aiccm_spin_state"]["per_cell"] == []


def test_runner_refusals(tmp_path):
    with pytest.raises(ValueError, match="aiccm_spin_state"):
        _job(tmp_path, "a", _li_doublet(), variant="four-center",
             aiccm_lattice_extension=(1, 1, 1), aiccm_spin_state="per-cell")
    with pytest.raises(ValueError, match="aiccm_spin_state"):
        _job(tmp_path, "b", _li_doublet(), method="UHF", jk_method="gdf",
             kpoints=(1, 1, 1), aiccm_spin_state="per-cell")
    with pytest.raises(ValueError, match="spin_state must be"):
        _job(tmp_path, "c", _li_doublet(), variant="real-gamma",
             aiccm_lattice_extension=(1, 1, 1), aiccm_spin_state="afm")
    # An explicit total that is not uniform per cell is refused by the
    # Bloch variant before any fit is built.
    with pytest.raises(ValueError, match="not uniform per cell"):
        _job(tmp_path, "d", _li_doublet(), variant="neutral-bloch",
             aiccm_lattice_extension=(2, 2, 1), aiccm_spin_state=3)


def test_runner_explicit_total_reaches_the_unrestricted_reference(tmp_path):
    """A singlet cell asked for a uniform triplet torus runs UHF."""
    res = _job(tmp_path, "trip", _h2_cube(), variant="real-gamma",
               aiccm_lattice_extension=(1, 1, 1), aiccm_spin_state=3)
    run = _run_fields(tmp_path / "trip")
    assert (run["aiccm_spin_state"]["n_alpha"], run["aiccm_spin_state"]["n_beta"]) == (2, 0)
    assert res.mo_energies_beta is not None
