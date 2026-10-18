"""M2 gate, rung 2a: the published pob cohesive energies through the objective.

The M2 gate (#224, ``vibe-basis/ROADMAP.md`` § 4) asks that
``CohesiveObjective``, evaluated at the shipped pob-TZVP-rev2, reproduce the
recorded statistics of pob against the plane-wave limit: **MD -24.6, MAD 35.4
kJ/mol over n = 28** (GPAW minus pob). It splits the way M0 gate 1 does in
``test_cohesive_reference_arithmetic.py``:

* **2a (this file)** -- given the *published* pob cohesive energies, does the
  objective reproduce the recorded statistics against ``pwref_merged.json``?
  This certifies the objective algebra, the reference file, the system mapping
  and the zero-point / counterpoise convention. It is exact and computes
  nothing, so a failure here is bookkeeping, never chemistry.
* **2b** -- does an *engine* reproduce those pob energies? That is the
  28-system campaign, M0 gate 1b on CRYSTAL23 and then ``VibeQcEngine``, and
  it is not this file's job. Per ROADMAP.md § 2 no phase is complete until it
  runs on ``VibeQcEngine``, so passing this file does not close the gate.

Provenance
----------
``studies/pw_limit_atomization/CONSOLIDATED.md`` (production table, n = 28):
the ``pob`` column is the published r2SCAN/pob-TZVP-rev2 cohesive energy, kJ/mol
per formula unit, from the reference set behind the pob basis-set papers
(``references-cohesive/r2scan.dat``, CRYSTAL23); ``Δ(G−pob)`` is GPAW minus pob.
Both are pinned here as constants because the reference tree is outside this
repository. The plane-wave side is read from ``pwref_merged.json`` itself,
which is the file the objective consumes in production.

Conventions, read off the reference decks (see the M0 test's docstring):

* **Static lattice on both sides.** The pob decks correct experiment upward by
  ZPE rather than theory downward; the GPAW atomization energies are
  static-lattice. So ``expect_zero_point=False``.
* **Counterpoise-corrected on the Gaussian side.** The pob free atoms use
  ``ATOMBSSE``, i.e. they sit in the crystal's ghost basis. A plane-wave
  basis has no superposition error, so the BSSE-free Gaussian number is the
  comparable one: ``expect_counterpoise=True``.

Sign: the objective's residual is ``candidate - reference`` (pob minus GPAW),
the opposite of the table's ``Δ(G−pob)``. So the objective must report
**MD +24.6**; the gate's -24.6 is the same statistic in the table's sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip(
    "vibe_basis.objective", reason="vibe-basis is an optional [basisopt] extra"
)

from vibe_basis.objective import (  # noqa: E402
    CohesiveObjective,
    ObjectiveError,
    load_pw_reference,
    split_validation,
)

_PWREF = (
    Path(__file__).resolve().parents[2]
    / "studies" / "pw_limit_atomization" / "runs" / "pwref_merged.json"
)

# system_id: (E_pob, published GPAW - pob), kJ/mol per formula unit,
# transcribed from CONSOLIDATED.md's production table.
_PUBLISHED = {
    "agcl_rocksalt": (539.7, -16.3),
    "alas_zincblende": (788.4, -43.5),
    "aln_zincblende": (1104.6, 10.1),
    "alp_zincblende": (816.4, -0.6),
    "bn_zincblende": (1300.0, 14.1),
    "bao_rocksalt": (1120.8, -131.5),
    "bas_rocksalt": (1099.3, -158.2),
    "c_diamond": (716.6, 7.2),
    "caf2_fluorite": (1609.9, -20.3),
    "cao_rocksalt": (1032.9, 56.3),
    "cdse_zincblende": (524.8, -29.3),
    "cscl_cscl": (678.5, -57.8),
    "gaas_zincblende": (697.9, -57.6),
    "gan_beta_zincblende": (876.6, -21.9),
    "gap_zincblende": (723.6, -25.5),
    "inas_zincblende": (623.3, -36.3),
    "kbr_rocksalt": (661.8, -68.2),
    "kcl_rocksalt": (690.1, -52.2),
    "lih_rocksalt": (459.1, 5.2),
    "li2o_antifluorite": (1122.2, 14.2),
    "licl_rocksalt": (688.1, -12.7),
    "lif_rocksalt": (843.7, -3.5),
    "mgo_rocksalt": (1044.3, -34.8),
    "mgs_rocksalt": (780.0, -7.9),
    "nacl_rocksalt": (662.8, -29.8),
    "naf_rocksalt": (761.8, -7.8),
    "sic_zincblende": (1190.3, 44.3),
    "zns_zincblende": (639.1, -24.0),
}

#: The gate's numbers, in the table's GPAW - pob sign.
GATE_MD_GPAW_MINUS_POB = -24.6
GATE_MAD = 35.4
GATE_N = 28

#: Every input is published to 0.1 kJ/mol, so each residual carries at most
#: 0.05 of rounding and so do the averages; the recorded statistics add their
#: own half-step. 0.1 kJ/mol covers both and nothing else.
ROUNDING_TOL_KJ = 0.1


@dataclass(frozen=True)
class _PublishedPob:
    """One published pob cohesive energy, flagged with the decks' conventions."""

    system: str
    cohesive_kj_per_mol: float
    ok: bool = True
    zero_point_applied: bool = False
    counterpoise_applied: bool = True


def _candidates():
    return [_PublishedPob(s, pob) for s, (pob, _) in _PUBLISHED.items()]


def _objective(**kwargs):
    return CohesiveObjective(
        load_pw_reference(_PWREF),
        loss="mad",
        expect_zero_point=False,
        expect_counterpoise=True,
        **kwargs,
    )


def test_the_published_table_and_the_reference_file_name_the_same_systems():
    assert set(load_pw_reference(_PWREF)) == set(_PUBLISHED)


def test_each_published_delta_is_reproduced_within_its_rounding():
    """Catches a mis-mapped system before it can hide inside an average."""
    refs = load_pw_reference(_PWREF)
    worst = {
        system: abs((refs[system] - pob) - delta)
        for system, (pob, delta) in _PUBLISHED.items()
    }
    bad = {s: d for s, d in worst.items() if d > 0.05 + 1e-9}
    assert not bad, f"GPAW - pob differs from the published column: {bad}"


def test_the_objective_reproduces_the_recorded_statistics():
    """The gate's two numbers, end to end through the shipped objective."""
    report = _objective().report(_candidates())
    train = report.train
    assert train.n == GATE_N
    assert -train.md == pytest.approx(GATE_MD_GPAW_MINUS_POB, abs=ROUNDING_TOL_KJ)
    assert train.mad == pytest.approx(GATE_MAD, abs=ROUNDING_TOL_KJ)
    assert report.missing == ()


def test_the_comparison_is_convention_checked():
    """Pinned conventions leave nothing unchecked; the report says so."""
    assert _objective().report(_candidates()).unchecked_conventions == ()


def test_a_zero_point_corrected_candidate_is_refused():
    """A ZPE-corrected number would fold the correction into the basis error."""
    candidates = [
        _PublishedPob(c.system, c.cohesive_kj_per_mol,
                      zero_point_applied=c.system == "mgo_rocksalt")
        for c in _candidates()
    ]
    with pytest.raises(ObjectiveError, match="zero-point"):
        _objective().report(candidates)


def test_an_uncorrected_candidate_is_refused():
    """Counterpoise-free Gaussian atoms carry BSSE the plane-wave side lacks."""
    candidates = [
        _PublishedPob(c.system, c.cohesive_kj_per_mol,
                      counterpoise_applied=c.system != "lih_rocksalt")
        for c in _candidates()
    ]
    with pytest.raises(ObjectiveError, match="counterpoise"):
        _objective().report(candidates)


def test_the_held_out_split_is_deterministic_and_reported():
    """The diagnostic a campaign quotes; pinned so two runs stay comparable."""
    systems = sorted(_PUBLISHED)
    held_out = split_validation(systems, every=4, offset=0)
    assert held_out == split_validation(systems, every=4, offset=0)
    report = _objective(validation=held_out).report(_candidates())
    assert report.train.n + report.validation.n == GATE_N
    assert report.validation.n == len(held_out) == 7
