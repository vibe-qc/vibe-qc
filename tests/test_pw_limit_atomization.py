"""Tests for the GPAW plane-wave-limit atomization reference generator.

The pure-Python tests (extrapolation, systems-data integrity, functional map)
run everywhere and are fast. The end-to-end test runs GPAW out-of-process and is
skipped when GPAW/ASE are not importable in a subprocess (CLAUDE.md §10: we
never import gpaw in-process, not even in tests).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from examples.regression.pw_limit_atomization import (
    PW_XC,
    VALIDATION_SET,
    compute_pw_atomization,
)
from examples.regression.pw_limit_atomization.pw_reference import (
    CutoffPoint,
    _extrapolate_pw_limit,
)
from examples.regression.pw_limit_atomization.systems import (
    BY_ID,
    HUND_MAGMOM,
    MAIN_GROUP_SET,
    NONCUBIC_SET,
    NONMAGNETIC_SET,
    TM_SYSTEMS,
    VALIDATION_SET,
)


def _cp(cutoff, atomization):
    return CutoffPoint(
        cutoff_ev=cutoff,
        e_solid_ha=0.0,
        e_atoms_ha=0.0,
        atomization_kjmol=atomization,
        solid_converged=True,
        atoms_converged=True,
    )


# --------------------------- pure-Python logic ---------------------------- #
def test_extrapolation_flat_series_returns_value():
    # a converged (flat) series must extrapolate to that value
    limit, how = _extrapolate_pw_limit([_cp(1000, 840.0), _cp(1200, 840.0)])
    assert limit == pytest.approx(840.0, abs=1e-9)
    assert "extrapolation" in how


def test_extrapolation_single_point_is_identity():
    limit, how = _extrapolate_pw_limit([_cp(800, 812.3)])
    assert limit == pytest.approx(812.3)
    assert "no extrapolation" in how


def test_extrapolation_rising_series_extends_past_top():
    # still rising with cutoff -> PW limit lies beyond the highest point
    limit, _ = _extrapolate_pw_limit([_cp(800, 830.0), _cp(1000, 838.0)])
    assert limit > 838.0


def test_functional_map_has_core_functionals():
    for f in ("lda", "pbe", "scan", "r2scan"):
        assert f in PW_XC
    assert PW_XC["r2scan"] == "MGGA_X_R2SCAN+MGGA_C_R2SCAN"


def test_systems_data_integrity():
    assert VALIDATION_SET and len(BY_ID) == len(NONMAGNETIC_SET) + len(
        NONCUBIC_SET
    ) + len(TM_SYSTEMS)
    assert len(NONMAGNETIC_SET) == len(VALIDATION_SET) + len(MAIN_GROUP_SET)
    for s in NONMAGNETIC_SET:
        assert s.n_fu_per_cell >= 1
        assert s.vasp900_kjmol > 0
        # every species must have a Hund magmom and appear in the composition
        for sym in s.atom_symbols():
            assert sym in HUND_MAGMOM
            assert isinstance(s.magmom(sym), int)
        # elemental solids ("C") carry per-atom == per-f.u. via n_fu_per_cell
        if len(s.composition) == 1:
            assert s.n_fu_per_cell >= 1


def test_unknown_functional_rejected():
    with pytest.raises(ValueError):
        compute_pw_atomization(BY_ID["lif_rocksalt"], functional="not_a_functional")


def test_atom_scf_fallback_settings_support_k_ca():
    # GPAW-PWREF-001 regression: the atom worker's SCF fallback must not cap
    # iterations at 150 and must carry the explicit 1e-6 eV^2/e eigenstates
    # criterion — K/Ca semicore meta-GGA atoms need >150 iterations and
    # plateau near 1e-7 on the default 4e-8 threshold (energy already stable
    # to 1e-6 eV).  The worker is an embedded source string executed only in
    # a subprocess (§10), so pin the settings textually.
    from examples.regression.pw_limit_atomization.pw_reference import _GPAW_WORKER

    assert "min(maxiter, 150)" not in _GPAW_WORKER
    assert '"eigenstates": 1e-6' in _GPAW_WORKER
    for sym in ("K", "Ca", "Br"):
        assert f'"{sym}"' in _GPAW_WORKER  # stays on the SCF fallback list
    # K keeps its damped-mixer branch (charge-sloshing excursions otherwise)
    assert "_SCF_SLOSHY" in _GPAW_WORKER
    assert '"density": 1.5e-3' in _GPAW_WORKER


# --------------------------- GPAW end-to-end ------------------------------ #
def _gpaw_available() -> bool:
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import gpaw, ase"], capture_output=True, timeout=120
        )
        return r.returncode == 0
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _gpaw_available(), reason="gpaw/ase not importable")
def test_pipeline_runs_lif_pbe_smoke():
    # cheap PBE smoke test: validates the out-of-process pipeline end to end
    # (subprocess boundary, solid+atom orchestration, formula-unit arithmetic,
    # valence read-back). Not a physics assertion — wide sanity band only.
    res = compute_pw_atomization(
        BY_ID["lif_rocksalt"],
        functional="pbe",
        cutoffs_ev=(400.0,),
        kmesh=(2, 2, 2),
        atom_box_ang=9.0,
        conv_energy_ha=1e-4,
        max_iter=300,
    )
    assert len(res.points) == 1
    assert 400.0 < res.pw_limit_kjmol < 1200.0  # LiF atomization sanity
    assert res.atom_valence.get("Li") == 1
    assert res.atom_valence.get("F") == 7
    assert res.code_version != "unknown"
