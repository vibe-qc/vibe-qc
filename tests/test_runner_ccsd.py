"""run_job integration for canonical CCSD / CCSD(T)."""

from __future__ import annotations

import json
import tomllib

import pytest
from vibeqc import Atom, InsufficientMemoryError, Molecule, run_job
from vibeqc import CISDOptions
from vibeqc.cc import CCSDOptions


def _h2() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )


def _oh() -> Molecule:
    # OH radical (doublet), Bohr.
    ang = 1.8897259886
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.108444 * ang]),
         Atom(1, [0.0, 0.0, -0.867550 * ang])],
        charge=0,
        multiplicity=2,
    )


def test_ccsd_run_job_auto_selects_reference(tmp_path):
    """run_job auto-selects the CCSD(T) reference from the molecule + flag.

    Closed-shell routes to RHF + the spin-adapted kernel; an open-shell
    molecule (multiplicity > 1) routes to UHF + the spin-orbital kernel by
    default; ``ccsd_reference="rohf"`` selects ROHF + the same spin-orbital
    kernel.  The ``Job:`` line in the .out records the selected reference, so
    this pins the dispatch end-to-end.
    """
    # cc-pVDZ (a basis with a registered RI aux; sto-3g has none, which
    # run_job surfaces as an honest NotImplementedError -- see
    # test_ccsd_run_job_honest_error_no_aux below).
    # closed shell -> RHF
    stem_r = tmp_path / "h2_rhf"
    rr = run_job(_h2(), basis="cc-pvdz", method="ccsd(t)", output=stem_r)
    assert rr.ccsd.converged
    assert rr.ccsd.e_t < 0.0
    out_r = stem_r.with_suffix(".out").read_text()
    assert "Job: RHF + CCSD(T)" in out_r
    assert "Frozen core orbitals = 0 (ORCA 6.1 published count-only default)" in out_r
    assert "Algorithm            = DF-CCSD(T)" in out_r
    assert "Density fitting       = on" in out_r
    assert "RI auxiliary basis    = cc-pvdz-ri" in out_r

    # open shell, default -> UHF
    stem_u = tmp_path / "oh_uhf"
    ru = run_job(_oh(), basis="cc-pvdz", method="ccsd(t)", output=stem_u)
    assert ru.ccsd.converged
    assert ru.ccsd.e_t < 0.0
    out_u = stem_u.with_suffix(".out").read_text()
    assert "Job: UHF + CCSD(T)" in out_u
    assert "Frozen core orbitals = 1 (ORCA 6.1 published count-only default)" in out_u

    # open shell, ccsd_reference="rohf" -> ROHF (spin-pure reference)
    stem_o = tmp_path / "oh_rohf"
    ro = run_job(_oh(), basis="cc-pvdz", method="ccsd(t)",
                 ccsd_reference="rohf", output=stem_o)
    assert ro.ccsd.converged
    assert ro.ccsd.e_t < 0.0
    out_o = stem_o.with_suffix(".out").read_text()
    assert "Job: ROHF + CCSD(T)" in out_o
    assert "Frozen core orbitals = 1 (ORCA 6.1 published count-only default)" in out_o


def test_ccsd_run_job_records_explicit_all_electron_convention(tmp_path):
    stem = tmp_path / "h2_explicit_ae"
    res = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd",
        ccsd_options=CCSDOptions(n_frozen_core=0),
        output=stem,
        structured_log=True,
    )

    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "Frozen core orbitals = 0 (explicit all-electron)" in out
    assert "Algorithm            = DF-CCSD" in out
    assert "RI auxiliary basis    = cc-pvdz-ri" in out

    records = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    ccsd_event = next(r for r in records if r.get("event") == "ccsd_converged")
    assert ccsd_event["n_frozen_core"] == 0
    assert ccsd_event["frozen_core_convention"] == "all-electron-explicit"
    assert ccsd_event["algorithm"] == "DF-CCSD"
    assert ccsd_event["density_fit"] is True
    assert ccsd_event["aux_basis"] == "cc-pvdz-ri"


def test_ccsd_run_job_records_explicit_attribute_assignment(tmp_path):
    """``opts.n_frozen_core = 0`` (attribute path) is explicit, not default.

    Regression for the 2026-07-29 output-truthfulness defect: an
    explicitly supplied all-electron request via attribute assignment
    printed as "(chemical-core default)" because only the constructor
    kwarg path marked ``_n_frozen_core_explicit``.  The write-through
    property on CCSDOptions now marks every user write.
    """
    opts = CCSDOptions()
    opts.n_frozen_core = 0
    stem = tmp_path / "h2_attr_ae"
    res = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd",
        ccsd_options=opts,
        output=stem,
        structured_log=True,
    )

    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "Frozen core orbitals = 0 (explicit all-electron)" in out

    records = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    ccsd_event = next(r for r in records if r.get("event") == "ccsd_converged")
    assert ccsd_event["n_frozen_core"] == 0
    assert ccsd_event["frozen_core_convention"] == "all-electron-explicit"


def test_ccsd_run_job_honest_error_no_aux():
    """run_job CCSD on a basis without a registered RI aux fails early.

    BUG 111: The DF-CCSD path requires an RI auxiliary basis; for an orbital
    basis with no bundled default (sto-3g), run_job must raise a clear
    ValueError *before* SCF -- no wasted convergence, no cryptic
    NotImplementedError after SCF.  The error message names the orbital basis
    and offers actionable alternatives.
    """
    with pytest.raises(ValueError, match="auxiliary basis"):
        run_job(_h2(), basis="sto-3g", method="ccsd(t)")


def test_ccsd_run_job_sto3g_error_mentions_basis_and_guidance():
    """The fail-early error for CCSD(T)/STO-3G names the basis and offers options."""
    with pytest.raises(ValueError) as exc_info:
        run_job(_h2(), basis="sto-3g", method="ccsd(t)")
    msg = str(exc_info.value)
    assert "'sto-3g'" in msg or "sto-3g" in msg
    assert "auxiliary basis" in msg.lower()
    assert "def2-SVP" in msg
    assert "CCSDOptions" in msg


def test_ccsd_run_job_sto3g_explicit_aux_proceeds(tmp_path):
    """CCSD(T)/STO-3G with an explicit aux_basis runs without aux-missing error."""
    opts = CCSDOptions(aux_basis="cc-pvdz-ri", compute_triples=False)
    stem = tmp_path / "h2_sto3g_explicit_aux"
    res = run_job(
        _h2(),
        basis="sto-3g",
        method="ccsd",
        ccsd_options=opts,
        output=stem,
    )
    assert res.ccsd.converged


def test_ccsd_run_job_sto3g_aux_basis_kwarg_proceeds(tmp_path):
    """CCSD/STO-3G with run_job(aux_basis=...) runs without aux-missing error.

    BUG 113: the kwarg used to be wired to the SCF options only, so the
    fail-early CC aux gate never saw it and a sto-3g job with an explicit
    aux_basis= kwarg refused for want of a registered default.
    """
    stem = tmp_path / "h2_sto3g_aux_kwarg"
    res = run_job(
        _h2(),
        basis="sto-3g",
        method="ccsd",
        aux_basis="cc-pvdz-ri",
        output=stem,
    )
    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "RI auxiliary basis    = cc-pvdz-ri" in out


def test_ccsd_run_job_aux_basis_kwarg_reaches_cc_routes(tmp_path):
    """run_job(aux_basis=...) selects the RI basis for DF-CCSD and LCCSD.

    BUG 113: the kwarg reached the SCF options only, so the correlated
    route silently auto-resolved its own auxiliary basis instead.
    """
    stem_c = tmp_path / "h2_ccsd_kwarg"
    res_c = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd",
        aux_basis="cc-pvdz-ri",
        output=stem_c,
    )
    assert res_c.ccsd.converged
    out_c = stem_c.with_suffix(".out").read_text()
    assert "RI auxiliary basis    = cc-pvdz-ri" in out_c

    stem_l = tmp_path / "h2_lccsd_kwarg"
    res_l = run_job(
        _h2(),
        basis="cc-pvdz",
        method="lccsd",
        aux_basis="cc-pvdz-ri",
        output=stem_l,
    )
    assert res_l.ccsd.converged
    out_l = stem_l.with_suffix(".out").read_text()
    assert "RI auxiliary basis    = cc-pvdz-ri" in out_l


def test_ccsd_run_job_aux_basis_kwarg_conflict_raises(tmp_path):
    """run_job(aux_basis=...) contradicting ccsd_options.aux_basis raises."""
    with pytest.raises(ValueError, match="contradicts"):
        run_job(
            _h2(),
            basis="cc-pvdz",
            method="ccsd",
            aux_basis="cc-pvdz-ri",
            ccsd_options=CCSDOptions(aux_basis="cc-pvtz-ri"),
            output=tmp_path / "h2_conflict",
        )


def test_ccsd_run_job_aux_basis_unresolvable_fails_loudly(tmp_path):
    """An unresolvable explicit aux_basis raises, not silently ignored."""
    with pytest.raises(RuntimeError, match="no shells loaded"):
        run_job(
            _h2(),
            basis="cc-pvdz",
            method="ccsd",
            aux_basis="no-such-basis-xyz",
            output=tmp_path / "h2_badaux",
        )


def test_ccsd_run_job_sto3g_density_fit_false_proceeds(tmp_path):
    """CCSD/STO-3G with density_fit=False proceeds (canonical 4-index route)."""
    opts = CCSDOptions(density_fit=False, compute_triples=False)
    stem = tmp_path / "h2_sto3g_nodf"
    res = run_job(
        _h2(),
        basis="sto-3g",
        method="ccsd",
        ccsd_options=opts,
        output=stem,
    )
    assert res.ccsd.converged


def test_ccsd_run_job_ccpvdz_proceeds_with_default_aux(tmp_path):
    """CCSD(T)/cc-pVDZ proceeds normally (has a registered RI aux default)."""
    stem = tmp_path / "h2_ccpvdz"
    res = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd(t)",
        output=stem,
    )
    assert res.ccsd.converged
    assert res.ccsd.e_t < 0.0
    out = stem.with_suffix(".out").read_text()
    assert "RI auxiliary basis" in out


@pytest.mark.parametrize("process_budget", [None, 256 * 1024**2])
def test_ccsd_run_job_rejects_impossible_triples_cap_before_scf(
    tmp_path,
    monkeypatch,
    process_budget,
):
    opts = CCSDOptions(
        triples_memory_mode="direct",
        requested_memory_bytes=1,
    )

    def fail_if_scf_runs(*args, **kwargs):
        raise AssertionError("SCF ran before the impossible triples cap was rejected")

    monkeypatch.setattr("vibeqc.runner.available_memory_bytes", lambda: 0)
    monkeypatch.setattr("vibeqc.runner.run_rhf", fail_if_scf_runs)
    with pytest.raises(InsufficientMemoryError, match="minimum modeled CCSD"):
        run_job(
            _h2(),
            basis="cc-pvdz",
            method="ccsd(t)",
            ccsd_options=opts,
            memory_budget_bytes=process_budget,
            output=tmp_path / "impossible_triples_cap",
        )


def test_ccsd_run_job_rejects_invalid_triples_mode_before_scf(
    tmp_path,
    monkeypatch,
):
    opts = CCSDOptions(triples_memory_mode="disk-backed")

    def fail_if_scf_runs(*args, **kwargs):
        raise AssertionError("SCF ran before the invalid triples mode was rejected")

    monkeypatch.setattr("vibeqc.runner.run_rhf", fail_if_scf_runs)
    with pytest.raises(ValueError, match="triples_memory_mode"):
        run_job(
            _h2(),
            basis="cc-pvdz",
            method="ccsd(t)",
            ccsd_options=opts,
            output=tmp_path / "invalid_triples_mode",
        )


def test_ccsd_run_job_direct_budget_telemetry_and_citation(tmp_path):
    cap = 16 * 1024**2
    opts = CCSDOptions(
        triples_memory_mode="direct",
        requested_memory_bytes=cap,
    )
    stem = tmp_path / "h2_direct_triples"
    result = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd(t)",
        ccsd_options=opts,
        memory_budget_bytes=256 * 1024**2,
        num_threads=1,
        output=stem,
        structured_log=True,
    )

    assert result.ccsd.triples_memory_mode_used == "direct"
    assert result.ccsd.triples_threads_used == 1
    assert result.ccsd.triples_workspace_bytes <= cap
    records = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    memory = next(r for r in records if r.get("event") == "memory_estimate")
    cc_event = next(r for r in records if r.get("event") == "ccsd_converged")
    assert memory["native_workspace_budget_bytes"] == cap
    assert cc_event["triples_memory_mode"] == "direct"
    assert cc_event["triples_workspace_bytes"] <= cap
    assert "10.1021/acs.jctc.9b00957" in stem.with_suffix(
        ".references"
    ).read_text()


def test_ccsd_run_job_disk_budget_telemetry_cleanup_and_citation(tmp_path):
    cap = 16 * 1024**2
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    opts = CCSDOptions(
        triples_memory_mode="disk",
        requested_memory_bytes=cap,
        triples_tile_size=1,
        triples_max_threads=1,
        triples_scratch_directory=str(scratch),
    )
    stem = tmp_path / "h2_disk_triples"
    result = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd(t)",
        ccsd_options=opts,
        memory_budget_bytes=256 * 1024**2,
        num_threads=1,
        output=stem,
        structured_log=True,
    )

    assert result.ccsd.triples_memory_mode_used == "disk"
    assert result.ccsd.triples_workspace_bytes <= cap
    assert result.ccsd.triples_disk_bytes > 0
    assert list(scratch.iterdir()) == []
    records = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    event = next(r for r in records if r.get("event") == "ccsd_converged")
    assert event["triples_memory_mode"] == "disk"
    assert event["triples_disk_bytes"] > 0
    assert "10.1021/acs.jctc.9b00957" in stem.with_suffix(
        ".references"
    ).read_text()


def test_uccsd_run_job_direct_budget_has_no_closed_shell_citation(tmp_path):
    cap = 64 * 1024**2
    opts = CCSDOptions(
        triples_memory_mode="direct",
        requested_memory_bytes=cap,
        triples_max_threads=1,
    )
    stem = tmp_path / "oh_direct_triples"
    result = run_job(
        _oh(),
        basis="cc-pvdz",
        method="ccsd(t)",
        ccsd_options=opts,
        memory_budget_bytes=256 * 1024**2,
        num_threads=1,
        output=stem,
        structured_log=True,
    )

    assert result.ccsd.triples_memory_mode_used == "direct"
    assert result.ccsd.triples_workspace_bytes <= cap
    event = next(
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "ccsd_converged"
    )
    assert event["triples_memory_mode"] == "direct"
    assert "10.1021/acs.jctc.9b00957" not in stem.with_suffix(
        ".references"
    ).read_text()


def test_uccsd_run_job_disk_fails_before_scf(tmp_path, monkeypatch):
    opts = CCSDOptions(
        triples_memory_mode="disk",
        requested_memory_bytes=64 * 1024**2,
    )

    def fail_if_scf_runs(*args, **kwargs):
        raise AssertionError("SCF ran before open-shell disk mode was rejected")

    monkeypatch.setattr("vibeqc.runner.run_uhf", fail_if_scf_runs)
    with pytest.raises(ValueError, match="cannot spill"):
        run_job(
            _oh(),
            basis="cc-pvdz",
            method="ccsd(t)",
            ccsd_options=opts,
            memory_budget_bytes=256 * 1024**2,
            output=tmp_path / "oh_disk_rejected",
        )


def test_ccsd_run_job_sto3g_fails_before_scf_output(tmp_path):
    """CCSD(T)/STO-3G fails before SCF produces any output file (BUG 111).

    The fix moves the aux check to before SCF, so no .out file should be
    written for a calculation that cannot proceed.  If this test fails,
    the check may have regressed to the post-SCF position.
    """
    stem = tmp_path / "h2_sto3g_noout"
    with pytest.raises(ValueError, match="auxiliary basis"):
        run_job(_h2(), basis="sto-3g", method="ccsd(t)", output=stem)
    # No .out file should exist because the calculation was aborted
    # before SCF began.
    assert not stem.with_suffix(".out").exists()


def test_ccsd_triples_selector_controls_run_job_label(tmp_path):
    stem = tmp_path / "h2_ccsd_t"
    res = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd",
        triples="(t)",
        output=stem,
        ccsd_options=CCSDOptions(triples="none"),
    )

    assert res.converged
    assert res.ccsd.converged
    assert res.energy_total == pytest.approx(res.ccsd.e_total, abs=1e-12)

    out = stem.with_suffix(".out").read_text()
    assert "Job: RHF + CCSD(T)" in out
    assert "Coupled-Cluster CCSD(T)" in out
    assert "E(T) correction" in out

    refs = stem.with_suffix(".references").read_text()
    assert "Raghavachari" in refs


def test_ccsd_triples_selector_can_disable_method_t(tmp_path):
    stem = tmp_path / "h2_ccsd"
    res = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd(t)",
        triples="none",
        output=stem,
    )

    assert res.converged
    assert res.ccsd.converged

    out = stem.with_suffix(".out").read_text()
    assert "Job: RHF + CCSD" in out
    assert "Job: RHF + CCSD(T)" not in out
    assert "Coupled-Cluster CCSD" in out
    assert "E(T) correction" not in out


def test_ccsd_triples_selector_runs_accsdt(tmp_path):
    res = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd",
        triples="A-CCSD(T)",
        output=tmp_path / "h2_accsd_t",
    )

    assert res.method == "a-ccsd(t)"
    assert res.ccsd.converged
    assert res.ccsd.lambda_residual_norm < 1e-7
    out = (tmp_path / "h2_accsd_t.out").read_text()
    assert "Coupled-Cluster A-CCSD(T)" in out
    assert "Algorithm            = DF-A-CCSD(T)" in out
    assert "E(A-T) correction" in out


def test_triples_selector_is_ccsd_only(tmp_path):
    with pytest.raises(ValueError, match="triples="):
        run_job(
            _h2(),
            basis="sto-3g",
            method="rhf",
            triples="none",
            output=tmp_path / "h2_rhf",
        )


def test_ccsd_options_are_cc_family_only(tmp_path):
    with pytest.raises(ValueError, match="ccsd_options"):
        run_job(
            _h2(),
            basis="sto-3g",
            method="rhf",
            ccsd_options=CCSDOptions(),
            output=tmp_path / "h2_rhf_opts",
        )


def test_citype_cisd_runs_fixed_space_ci(tmp_path):
    stem = tmp_path / "h2_cisd"
    res = run_job(
        _h2(),
        basis="sto-3g",
        method="ci",
        citype="cisd",
        cisd_options=CISDOptions(nroots=2),
        output=stem,
    )

    assert res.converged
    assert res.method.startswith("cisd(")
    assert res.energy == pytest.approx(res.energy_trace[-1], abs=1e-12)

    out = stem.with_suffix(".out").read_text()
    assert "Job: RHF + CISD" in out
    assert "Method:            cisd(" in out
    assert "Determinants:" in out
    assert "root 1:" in out

    refs = stem.with_suffix(".references").read_text()
    assert "Pople" in refs


def test_citype_ccsd_t_aliases_existing_path(tmp_path):
    stem = tmp_path / "h2_citype_ccsd_t"
    res = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ci",
        citype="ccsd(t)",
        output=stem,
    )

    assert res.converged
    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "Job: RHF + CCSD(T)" in out


def test_citype_qcisd_routes(tmp_path):
    stem = tmp_path / "h2_qcisd"
    res = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ci",
        citype="qcisd",
        output=stem,
    )

    assert res.converged
    assert res.ccsd.converged
    assert res.method == "qcisd"
    out = stem.with_suffix(".out").read_text()
    assert "Job: RHF + QCISD" in out
    assert "Coupled-Pair / Coupled-Cluster QCISD" in out


def test_ccsd_run_job_manifest_declares_integral_route(tmp_path):
    """#23 limb (b): the .system manifest names the CC integral route.

    Default CCSD(T) is density-fitted and CCSDOptions(density_fit=False)
    selects the canonical four-index route; the L7 convention gate must
    be able to refuse a DF-vs-conventional pair from the manifest alone.
    Both routes therefore declare the executed integral route in the
    ``[run]`` section, with exact values (a conventional route records
    exactly False / exactly "", never an approximate "off").
    """
    stem_df = tmp_path / "h2_df"
    run_job(_h2(), basis="cc-pvdz", method="ccsd(t)", output=stem_df)
    with stem_df.with_suffix(".system").open("rb") as f:
        body_df = tomllib.load(f)
    run_df = body_df["run"]
    assert run_df.get("cc_density_fit") is True
    assert run_df.get("cc_auxiliary_basis") == "cc-pvdz-ri"
    assert run_df.get("cc_algorithm") == "DF-CCSD(T)"

    # Negative control (L125): the SAME route with the feature off.
    stem_cv = tmp_path / "h2_cv"
    run_job(
        _h2(),
        basis="cc-pvdz",
        method="ccsd(t)",
        ccsd_options=CCSDOptions(density_fit=False),
        output=stem_cv,
    )
    with stem_cv.with_suffix(".system").open("rb") as f:
        body_cv = tomllib.load(f)
    run_cv = body_cv["run"]
    assert run_cv.get("cc_density_fit") is False
    assert run_cv.get("cc_auxiliary_basis") == ""
    assert run_cv.get("cc_algorithm") == "CCSD(T)"


def test_citype_qcisd_t_routes(tmp_path):
    stem = tmp_path / "h2_qcisd_t"
    res = run_job(
        _h2(),
        basis="cc-pvdz",
        method="ci",
        citype="qcisd(t)",
        output=stem,
    )

    assert res.converged
    assert res.ccsd.converged
    assert res.method == "qcisd(t)"
    out = stem.with_suffix(".out").read_text()
    assert "Job: RHF + QCISD(T)" in out
    assert "E(QCISD(T))" in out
