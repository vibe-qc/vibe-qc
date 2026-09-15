"""Composite 3c methods registry — framework tests.

Covers :mod:`vibeqc.composites`. The tests here exercise the catalogue and
dispatcher contracts, with focused end-to-end SCF coverage in
``test_composite_integration.py`` and ``test_xc_rsh.py``.

For each recipe the assertion shape is:

* The recipe is registered and resolvable by both ASCII and unicode
  spellings (``"wb97x-3c"`` / ``"ωb97x-3c"``).
* The recipe carries a sensible (basis, dispersion, gcp_basis) tuple
  and an availability flag.
* Composites flagged ``PENDING_F1`` / ``PENDING_F3`` raise
  :class:`CompositeUnavailable` on ``run_job(method="...")``.
* The catalogue is stable: every recipe has a non-empty citation,
  every basis name shows up in the :mod:`vibeqc.gcp` registry,
  every gCP-using recipe references a basis whose gCP params exist.

Two end-to-end SCF cells live behind ``@pytest.mark.composite_full``,
gated on the gCP per-element data being present for the relevant
elements at the relevant basis — currently skip-by-default, will
auto-enable as the data tables land.
"""

from __future__ import annotations

import pytest

import vibeqc as vq
from vibeqc.composites import Availability, ShortRangeCorrection
from vibeqc.gcp import compute_gcp


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

EXPECTED_NAMES = {
    "hf-3c", "pbeh-3c", "b97-3c", "b3lyp-3c",
    "r2scan-3c", "wb97x-3c", "hse-3c",
}


def test_all_seven_composites_registered():
    names = {r.name for r in vq.list_composites()}
    assert names == EXPECTED_NAMES


def test_resolve_composite_is_case_insensitive():
    a = vq.resolve_composite("HF-3c")
    b = vq.resolve_composite("hf-3c")
    c = vq.resolve_composite("hf-3C")
    assert a is b is c is not None


def test_resolve_composite_accepts_unicode_omega():
    a = vq.resolve_composite("wb97x-3c")
    b = vq.resolve_composite("ωB97X-3c")
    assert a is b is not None


def test_resolve_composite_accepts_superscript_two():
    a = vq.resolve_composite("r2scan-3c")
    b = vq.resolve_composite("r²scan-3c")
    assert a is b is not None


def test_resolve_composite_unknown_returns_none():
    assert vq.resolve_composite("totally-fake-3c") is None


def test_every_recipe_has_citation_and_notes():
    for r in vq.list_composites():
        assert r.citation, f"{r.name}: empty citation"
        assert r.notes, f"{r.name}: empty notes"


def test_every_recipe_has_known_gcp_basis():
    """Every recipe that references gCP should name a basis whose
    parameters exist in the gCP registry. (The per-element data may
    still be PENDING_GCP_DATA; that's checked at evaluation time, not
    at registration time.)
    """
    for r in vq.list_composites():
        if r.gcp_basis is None:
            continue
        params = vq.gcp_params_for(r.gcp_basis)
        assert params is not None, (
            f"{r.name}: gCP basis {r.gcp_basis!r} has no params"
        )


def test_dispersion_choice_is_valid():
    for r in vq.list_composites():
        assert r.dispersion in ("d3bj", "d4", "none"), (
            f"{r.name}: invalid dispersion {r.dispersion!r}"
        )


def test_availability_values_are_enum_members():
    for r in vq.list_composites():
        assert isinstance(r.availability, Availability)


# ---------------------------------------------------------------------------
# Per-recipe expected wiring
# ---------------------------------------------------------------------------

def test_hf_3c_recipe():
    r = vq.resolve_composite("hf-3c")
    assert r is not None
    assert r.functional is None       # pure HF
    assert r.basis == "minix"
    assert r.dispersion == "d3bj"
    assert r.sr_mod is not None       # HF-3c carries the SRB
    assert isinstance(r.sr_mod, ShortRangeCorrection)


def test_r2scan_3c_recipe_uses_canonical_mctc_gcp_params():
    """Post-v0.9.0 bundling: r²SCAN-3c uses the canonical mctc-gcp
    parameters (σ, η, α, β) = (1.0, 1.315, 0.941, 1.4636) per
    Grimme 2021 + the mctc-gcp Fortran reference. The earlier
    "zero by design" assumption was wrong — gCP is nonzero for
    r²SCAN-3c but uses an etaspec=1.15 heavy-element scaling."""
    p = vq.gcp_params_for("def2-mtzvpp")
    assert p is not None
    assert abs(p.sigma - 1.0) < 1e-12
    assert abs(p.eta - 1.315) < 1e-12
    assert abs(p.alpha - 0.941) < 1e-12
    assert abs(p.beta - 1.4636) < 1e-12


def test_wb97x_3c_recipe_is_runnable_with_inline_ecp():
    """ωB97X-3c uses vDZP's sidecar ECPs through the inline primitive feed."""
    r = vq.resolve_composite("wb97x-3c")
    assert r is not None
    assert r.availability == Availability.RUNNABLE
    assert r.functional == "wb97x-v"
    assert r.basis == "vdzp"
    assert r.dispersion == "d4"


def test_r2scan_3c_recipe_is_now_runnable():
    """F3 (mGGA τ-density support) + def2-mTZVPP basis bundling both
    landed in v0.9.0. r²SCAN-3c runs SCF + D4 + damped gCP end-to-end.
    (Its gCP is NOT zero — the earlier 'zero by design' claim was wrong;
    audit F1.5. It carries the damped form, gcp_damping set.)"""
    r = vq.resolve_composite("r2scan-3c")
    assert r is not None
    assert r.availability == Availability.RUNNABLE
    assert r.gcp_damping is not None
    assert r.gcp_basis == "def2-mtzvpp"


def test_b97_3c_carries_srb_correction():
    r = vq.resolve_composite("b97-3c")
    assert r is not None
    assert r.sr_mod is not None
    assert r.sr_mod.qscal != 0.0
    assert r.sr_mod.kind == "b973c_mod"


def test_pbeh_3c_has_no_srb():
    """PBEh-3c's basis contraction is tuned to make SRB unnecessary —
    only HF-3c and B97-3c carry SRB."""
    r = vq.resolve_composite("pbeh-3c")
    assert r is not None
    assert r.sr_mod is None


# ---------------------------------------------------------------------------
# Short-range correction evaluator
# ---------------------------------------------------------------------------

def test_srb_evaluator_returns_zero_for_single_atom():
    srb = ShortRangeCorrection(
        name="test", kind="hf3c_base", qscal=0.03, rscal=0.7, citation="test")
    assert srb.evaluate([8], [[0.0, 0.0, 0.0]]) == 0.0


def test_srb_evaluator_is_negative_for_bonded_pair():
    """SRB is binding — negative contribution at typical bond
    distances."""
    srb = ShortRangeCorrection(
        name="test", kind="hf3c_base", qscal=0.03, rscal=0.7, citation="test")
    # H-H at ~0.74 Å = 1.4 bohr
    e = srb.evaluate([1, 1], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    assert e < 0.0


def test_srb_evaluator_symmetric_in_atom_order():
    """Swapping atom order should not change E_SRB (the (Z_a · Z_b)^0.5
    weight and the |R_ab| distance are both symmetric)."""
    srb = ShortRangeCorrection(
        name="test", kind="hf3c_base", qscal=0.03, rscal=0.7, citation="test")
    e_ab = srb.evaluate(
        [6, 8], [[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]])
    e_ba = srb.evaluate(
        [8, 6], [[2.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert e_ab == pytest.approx(e_ba, abs=1e-12)


# ---------------------------------------------------------------------------
# run_job dispatcher gating
# ---------------------------------------------------------------------------

def test_no_composites_still_pending_f1():
    """Post-v0.9.0 F1 K-build wiring: no composite carries PENDING_F1
    any more. HSE-3c and ωB97X-3c are RUNNABLE; see test_xc_rsh.py for
    the inline-vDZP ECP route test."""
    pending = [r for r in vq.list_composites()
               if r.availability == Availability.PENDING_F1]
    assert pending == [], (
        f"Expected no PENDING_F1 composites, found: "
        f"{[r.name for r in pending]}"
    )


def test_run_job_r2scan_3c_runs_end_to_end(tmp_path):
    """Post-v0.9.0: F3 (mGGA) + def2-mTZVPP basis + D4 dispersion all
    landed; r²SCAN-3c is turnkey (SCF + D4 + damped gCP). SCF should
    converge on H2 and produce a sensible energy with the corrections
    applied."""
    if not vq.dftd4_available():
        pytest.skip("dftd4 not installed — r²SCAN-3c needs it for D4 dispersion")
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [1.4, 0.0, 0.0]),
    ], 0, 1)
    r = vq.run_job(
        mol, method="r2scan-3c",
        output=str(tmp_path / "r2_h2"), progress=False,
    )
    assert r.converged
    # r²SCAN/H2/def2-mTZVPP energy ~-1.169 Ha; D4 disp ~-1e-7 on H2.
    assert -1.20 < r.energy < -1.15


# ---------------------------------------------------------------------------
# Canonical SRB + gCP component values vs the grimme-lab/gcp Fortran
# reference (audit F1.4 / F1.5). Reference numbers were produced by
# compiling and running mctc-gcp (gcp.f90: basegrad, srb_egrad2,
# gcp_call) on these exact geometries. Component-level (no SCF), so the
# tolerances are tight; these would have caught both the wrong SRB form
# (F1.4) and the wrong gCP n_virt / missing damping (F1.5).
# ---------------------------------------------------------------------------

_H2O_Z = [8, 1, 1]
_H2O_R = [[0.0, 0.0, 0.0], [1.43, 0.0, 1.10], [-1.43, 0.0, 1.10]]
_H2_Z = [1, 1]
_H2_R = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]


def _h2o_mol():
    return vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0]),
                        vq.Atom(1, [1.43, 0.0, 1.10]),
                        vq.Atom(1, [-1.43, 0.0, 1.10])], 0, 1)


def test_hf3c_srb_matches_mctc_gcp_reference():
    srb = vq.resolve_composite("hf-3c").sr_mod
    assert srb.kind == "hf3c_base"
    assert srb.evaluate(_H2O_Z, _H2O_R) == pytest.approx(-0.035456180, abs=1e-7)
    assert srb.evaluate(_H2_Z, _H2_R) == pytest.approx(-0.001759739, abs=1e-7)


def test_b973c_srb_matches_mctc_gcp_reference():
    srb = vq.resolve_composite("b97-3c").sr_mod
    assert srb.kind == "b973c_mod"
    assert srb.evaluate(_H2O_Z, _H2O_R) == pytest.approx(-0.005713243, abs=1e-7)
    assert srb.evaluate(_H2_Z, _H2_R) == pytest.approx(-0.002683770, abs=1e-7)


def test_r2scan3c_gcp_matches_mctc_gcp_reference():
    """Damped gCP + r2scan3c virtual counts => +1.12 kcal/mol on H2O,
    matching mctc-gcp gcp_call('r2scan3c'). NOT zero (audit F1.5)."""
    r = vq.resolve_composite("r2scan-3c")
    d = (r.gcp_damping.dmp_scal, r.gcp_damping.dmp_exp)
    e = compute_gcp(_h2o_mol(), r.gcp_basis, damping=d).energy
    assert e == pytest.approx(0.001787149, abs=2e-6)


def test_pbeh3c_gcp_matches_mctc_gcp_reference():
    r = vq.resolve_composite("pbeh-3c")
    d = (r.gcp_damping.dmp_scal, r.gcp_damping.dmp_exp)
    e = compute_gcp(_h2o_mol(), r.gcp_basis, damping=d).energy
    assert e == pytest.approx(0.001314432, abs=2e-6)


def test_hf3c_gcp_matches_mctc_gcp_reference():
    """HF-3c MINIX gCP is undamped: +18.33 kcal/mol on H2O (= mctc-gcp
    gcp_call('hf3c') minus its SRB term). Guards the minix n_virt fix."""
    r = vq.resolve_composite("hf-3c")
    assert r.gcp_damping is None
    e = compute_gcp(_h2o_mol(), r.gcp_basis).energy
    assert e == pytest.approx(0.0292067, abs=1e-5)


def test_b973c_has_no_gcp():
    """B97-3c uses SRB only — mctc-gcp computes SRB-only for b973c, no
    gCP (audit F1.5)."""
    assert vq.resolve_composite("b97-3c").gcp_basis is None


def test_damped_gcp_suppresses_bonded_pairs():
    """Damping must make a single bonded molecule's gCP far smaller than
    the undamped value (the bonded-pair gCP is spurious — that is the
    whole point of the damped form; audit F1.5)."""
    r = vq.resolve_composite("r2scan-3c")
    mol = _h2o_mol()
    undamped = compute_gcp(mol, r.gcp_basis).energy
    damped = compute_gcp(
        mol, r.gcp_basis,
        damping=(r.gcp_damping.dmp_scal, r.gcp_damping.dmp_exp)).energy
    assert undamped > 0.02            # ~+28.8 kcal/mol undamped
    assert damped < 0.1 * undamped    # damping suppresses ~25x on H2O
