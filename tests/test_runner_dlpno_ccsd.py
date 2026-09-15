"""run_job integration for method="dlpno-ccsd" / "dlpno-ccsd(t)".

M5b wired both to the O(N⁶) pilot; M3c re-wired both ``dlpno-ccsd`` and
``dlpno-ccsd(t)`` to the reduced-scaling local solver
(`dlpno.ccsd_local_solver`), the latter with the local DLPNO-(T)
(`dlpno.triples_local`) on the converged amplitudes. The O(N⁶) pilot is
still reachable by explicitly passing a ``DLPNOCCSDPilotOptions``.
"""

from __future__ import annotations

import json
import tomllib

import pytest
from vibeqc import run_job
from vibeqc._vibeqc_core import Atom, Molecule


def _h2():
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )


class TestRunJobDLPNOCCSD:
    def test_ccsd_t_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = run_job(_h2(), basis="def2-svp", method="dlpno-ccsd(t)", output="h2")

        assert res.converged
        r = res.dlpno_ccsd
        assert r.converged
        assert r.triples_executed
        # Two electrons: (T) vanishes (the lone degenerate triple cancels to
        # floating-point noise in the spatial formulation); CCSD = FCI corr.
        assert abs(r.e_t) < 1e-12
        assert -0.05 < r.e_corr < -0.02
        assert res.energy_total == pytest.approx(r.e_total, abs=1e-12)

        out = (tmp_path / "h2.out").read_text()
        # The default closed-shell path is the rotated-occupied compatibility
        # correction; exact == canonical (T) at full domains.
        assert "DLPNO-CCSD(T) local reduced-scaling (M3c)" in out
        assert "E((T) correction)" in out
        assert "rotated-occupied (T1) compatibility correction" in out
        assert "triples_mode='t1'" in out

        refs = (tmp_path / "h2.references").read_text()
        assert "Riplinger" in refs
        assert "Raghavachari" in refs
        bib = (tmp_path / "h2.bibtex").read_text()
        assert "riplinger_dlpno_2013" in bib
        assert "guo_dlpno_t1_2018" not in bib

    def test_all_screened_triples_are_not_claimed(self, tmp_path, monkeypatch):
        """All artifacts report that an all-screened (T1) request skipped (T)."""
        from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

        monkeypatch.chdir(tmp_path)
        res = run_job(
            _h2(),
            basis="def2-svp",
            method="dlpno-ccsd(t)",
            output="screened",
            structured_log=True,
            dlpno_ccsd_options=LocalCCSDOptions(tcut_pairs=1.0),
        )

        assert res.dlpno_ccsd.converged
        assert res.dlpno_ccsd.n_pairs == 0
        assert res.dlpno_ccsd.n_screened > 0
        assert not res.dlpno_ccsd.triples_executed
        assert res.dlpno_ccsd.e_t == 0.0

        out = (tmp_path / "screened.out").read_text()
        assert "not executed for this result (triples_mode='none')" in out
        manifest = tomllib.loads((tmp_path / "screened.system").read_text())[
            "run"
        ]
        assert manifest["dlpno_triples_mode"] == "none"
        assert (
            manifest["dlpno_triples_algorithm"]
            == "not executed for this result"
        )

        records = [
            json.loads(line)
            for line in (tmp_path / "screened.scf.jsonl").read_text().splitlines()
            if line.strip()
        ]
        done = next(row for row in records if row["event"] == "dlpno_ccsd_done")
        assert done["triples_mode"] == "none"
        assert done["triples_algorithm"] == "not executed for this result"
        assert "guo_dlpno_t1_2018" not in (
            tmp_path / "screened.bibtex"
        ).read_text()

    def test_ccsd_uses_local_reduced_scaling_solver(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = run_job(
            _h2(),
            basis="def2-svp",
            method="dlpno-ccsd",
            output="h2c",
            structured_log=True,
        )
        r = res.dlpno_ccsd
        assert r.converged
        assert r.e_t == 0.0
        # H2 is two electrons ⇒ CCSD = FCI; the local solver must hit it.
        assert -0.05 < r.e_corr < -0.02
        out = (tmp_path / "h2c.out").read_text()
        assert "DLPNO-CCSD local reduced-scaling (M3c)" in out
        assert "canonical CCSD" in out  # full-domain == canonical CCSD note
        assert "Residual domain" in out and "= extended" in out
        assert "E((T) correction)" not in out

        manifest = tomllib.loads((tmp_path / "h2c.system").read_text())
        assert manifest["run"]["dlpno_residual_domain"] == "extended"
        records = [
            json.loads(line)
            for line in (tmp_path / "h2c.scf.jsonl").read_text().splitlines()
            if line.strip()
        ]
        done = next(r for r in records if r["event"] == "dlpno_ccsd_done")
        assert done["residual_domain"] == "extended"

    def test_aux_basis_kwarg_reaches_route(self, tmp_path, monkeypatch):
        """run_job(aux_basis=...) selects the DLPNO-CCSD RI basis (BUG 113)."""
        monkeypatch.chdir(tmp_path)
        res = run_job(
            _h2(),
            basis="def2-svp",
            method="dlpno-ccsd",
            output="h2aux",
            aux_basis="cc-pvdz-ri",
        )
        assert res.dlpno_ccsd.converged
        out = (tmp_path / "h2aux.out").read_text()
        assert "(RI: cc-pvdz-ri)" in out
        assert "RI: def2-svp-rifit" not in out

    def test_aux_basis_kwarg_conflict_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

        with pytest.raises(ValueError, match="contradicts"):
            run_job(
                _h2(),
                basis="def2-svp",
                method="dlpno-ccsd",
                output="h2conflict",
                aux_basis="cc-pvdz-ri",
                dlpno_ccsd_options=LocalCCSDOptions(aux_basis="cc-pvtz-ri"),
            )

    def _oh(self):
        return Molecule(
            [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.832])],
            charge=0,
            multiplicity=2,
        )

    def test_open_shell_routes_to_uccsd_t(self, tmp_path, monkeypatch):
        # method='dlpno-ccsd(t)' on an open-shell reference auto-routes to the
        # UHF-based spin-orbital DLPNO-UCCSD(T) local solver (M3c); the
        # O(N^6) pilot is opt-in via DLPNOUCCSDPilotOptions.
        monkeypatch.chdir(tmp_path)
        res = run_job(
            self._oh(), basis="def2-svp", method="dlpno-ccsd(t)", output="oh"
        )
        assert res.converged
        assert hasattr(res, "dlpno_ccsd")
        r = res.dlpno_ccsd
        assert r.converged
        # OH/def2-svp DLPNO-UCCSD at the default tcut_pno (~99.8% of the
        # -0.16523 anchor; see tests/test_dlpno_uccsd.py for the exactness gate).
        assert -0.17 < r.e_corr < -0.16
        assert r.e_t < 0.0
        assert res.energy_total == pytest.approx(r.e_total, abs=1e-12)

        out = (tmp_path / "oh.out").read_text()
        assert "DLPNO-UCCSD(T) local reduced-scaling" in out
        assert "E((T) correction)" in out
        assert "E(UCCSD correlation)" in out
        assert "DLPNO-(T0)" in out
        assert "triples_mode='t0'" in out

        # Citation surface: the open-shell paper must reach the references.
        refs = (tmp_path / "oh.references").read_text()
        assert "Saitow" in refs
        assert "Raghavachari" in refs  # the (T) correction

    def test_open_shell_guo_citation_tracks_actual_contraction(
        self, tmp_path, monkeypatch
    ):
        """Guo reaches every citation artifact only for a real Eq. (2) run."""
        from vibeqc.dlpno.uccsd_local_solver import LocalUCCSDOptions

        monkeypatch.chdir(tmp_path)
        common = dict(
            n_frozen=0,
            localise="boys",
            tcut_pno=0.0,
            tcut_mkn=0.0,
            tcut_pno_singles=0.0,
            coupling_radius=0.0,
            compute_triples=True,
            triples_mode="t1-iterative",
            aux_basis="cc-pvdz-ri",
        )
        executed = run_job(
            self._oh(),
            basis="sto-3g",
            method="dlpno-ccsd(t)",
            output="guo-executed",
            structured_log=True,
            dlpno_ccsd_options=LocalUCCSDOptions(
                tcut_pairs=0.0,
                **common,
            ),
        )
        assert executed.dlpno_ccsd.triples_executed
        assert "guo_dlpno_t1_2018" in (
            tmp_path / "guo-executed.bibtex"
        ).read_text()
        assert "Guo" in (tmp_path / "guo-executed.references").read_text()
        executed_out = (tmp_path / "guo-executed.out").read_text()
        algorithm = (
            "experimental dense spin-orbital generalization of Guo Eq. (2), "
            "iterative local-basis DLPNO-(T1)"
        )
        assert algorithm in executed_out
        assert "triples_mode='t1-iterative'" in executed_out
        assert (
            "DLPNO-UCCSD(T) local CCSD (M3c) + experimental dense "
            "iterative triples"
        ) in executed_out
        assert "The triples phase is dense and max_nbf-capped" in executed_out
        executed_records = [
            json.loads(line)
            for line in (
                tmp_path / "guo-executed.scf.jsonl"
            ).read_text().splitlines()
            if line.strip()
        ]
        executed_done = next(
            row for row in executed_records
            if row["event"] == "dlpno_uccsd_done"
        )
        assert executed_done["triples_mode"] == "t1-iterative"
        assert executed_done["triples_algorithm"] == algorithm
        assert executed_done["engine"] == (
            "local CCSD (M3c) + experimental dense iterative triples"
        )
        manifest = tomllib.loads(
            (tmp_path / "guo-executed.system").read_text()
        )
        keys = {row["key"] for row in manifest["citations"]["entries"]}
        assert "guo_dlpno_t1_2018" in keys
        assert "guo_openshell_dlpno_triples_2020" in keys

        skipped = run_job(
            self._oh(),
            basis="sto-3g",
            method="dlpno-ccsd(t)",
            output="guo-skipped",
            structured_log=True,
            dlpno_ccsd_options=LocalUCCSDOptions(
                tcut_pairs=1e-14,
                **common,
            ),
        )
        assert not skipped.dlpno_ccsd.triples_executed
        assert "guo_dlpno_t1_2018" not in (
            tmp_path / "guo-skipped.bibtex"
        ).read_text()
        assert "Guo" not in (tmp_path / "guo-skipped.references").read_text()
        skipped_out = (tmp_path / "guo-skipped.out").read_text()
        assert algorithm not in skipped_out
        assert "experimental dense iterative triples" not in skipped_out
        assert "not executed for this result (triples_mode='none')" in skipped_out
        skipped_records = [
            json.loads(line)
            for line in (
                tmp_path / "guo-skipped.scf.jsonl"
            ).read_text().splitlines()
            if line.strip()
        ]
        skipped_done = next(
            row for row in skipped_records
            if row["event"] == "dlpno_uccsd_done"
        )
        assert skipped_done["triples_mode"] == "none"
        assert skipped_done["triples_algorithm"] == (
            "not executed for this result"
        )
        assert skipped_done["engine"] == "local reduced-scaling (M3c)"
        manifest = tomllib.loads(
            (tmp_path / "guo-skipped.system").read_text()
        )
        keys = {row["key"] for row in manifest["citations"]["entries"]}
        assert "guo_dlpno_t1_2018" not in keys
        assert "guo_openshell_dlpno_triples_2020" not in keys

    def test_open_shell_ccsd_no_triples(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = run_job(
            self._oh(),
            basis="def2-svp",
            method="dlpno-ccsd",
            output="ohc",
            structured_log=True,
        )
        r = res.dlpno_ccsd
        assert r.converged
        assert r.e_t == 0.0
        out = (tmp_path / "ohc.out").read_text()
        assert "DLPNO-UCCSD local reduced-scaling" in out
        assert "E((T) correction)" not in out
        assert "Threshold convention" in out and "NormalPNO" in out
        assert "tcut_pairs 0.0001" in out
        assert "tcut_pno 3.33e-07" in out
        assert "tcut_mkn 0.001" in out
        assert "tcut_tno" not in out

        manifest = tomllib.loads((tmp_path / "ohc.system").read_text())
        assert manifest["run"]["dlpno_tcut_tno"] == ""
        records = [
            json.loads(line)
            for line in (tmp_path / "ohc.scf.jsonl").read_text().splitlines()
            if line.strip()
        ]
        done = next(r for r in records if r["event"] == "dlpno_uccsd_done")
        assert "tcut_tno" not in done["threshold_settings"]

    def test_open_shell_options_passed_through(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions

        # A coarse tcut_pno keeps this passthrough check cheap (heavy PNO
        # truncation -> small per-iteration cost); we only assert the options
        # reached the pilot, not the energy.
        res = run_job(
            self._oh(),
            basis="def2-svp",
            method="dlpno-ccsd",
            output="ohb",
            dlpno_ccsd_options=DLPNOUCCSDPilotOptions(
                localise="boys", n_frozen=1, tcut_pno=1e-4
            ),
        )
        r = res.dlpno_ccsd
        assert r.localise == "boys" and r.n_frozen == 1

    def test_open_shell_pilot_capped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions

        with pytest.raises(ValueError, match="max_nbf"):
            run_job(
                self._oh(),
                basis="def2-svp",
                method="dlpno-ccsd",
                output="ohguard",
                dlpno_ccsd_options=DLPNOUCCSDPilotOptions(max_nbf=3),
            )

    def test_local_solver_size_guard(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

        with pytest.raises(ValueError, match="max_nbf"):
            run_job(
                _h2(),
                basis="def2-svp",
                method="dlpno-ccsd",
                output="guard",
                dlpno_ccsd_options=LocalCCSDOptions(max_nbf=5),
            )

    def test_ccsd_t_still_pilot_capped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.ccsd import DLPNOCCSDPilotOptions

        with pytest.raises(ValueError, match="max_nbf"):
            run_job(
                _h2(),
                basis="def2-svp",
                method="dlpno-ccsd(t)",
                output="guardt",
                dlpno_ccsd_options=DLPNOCCSDPilotOptions(max_nbf=5),
            )

    def test_closed_shell_pilot_named_set_provenance(
        self, tmp_path, monkeypatch
    ):
        """The opt-in pilot discloses its supported TightPNO subset."""

        from vibeqc.dlpno import options_from_dlpno_thresholds
        from vibeqc.dlpno.ccsd import DLPNOCCSDPilotOptions

        monkeypatch.chdir(tmp_path)
        options = options_from_dlpno_thresholds(
            DLPNOCCSDPilotOptions, "tight"
        )
        run_job(
            _h2(),
            basis="def2-svp",
            method="dlpno-ccsd",
            output="pilot-tight",
            dlpno_ccsd_options=options,
            frozen_core=False,
            structured_log=True,
        )

        out = (tmp_path / "pilot-tight.out").read_text()
        assert "pilot (Riplinger 2013; O(N^6)" in out
        assert "TightPNO (supported subset)" in out
        assert "tcut_pno 1e-07" in out
        assert "tcut_mkn 0.0001" in out
        assert "tcut_pairs 1e-05" not in out
        assert "Unsupported cutoffs" in out and "tcut_pairs" in out
        assert "Frozen core orbitals = 0 (explicit all-electron)" in out
        assert "Residual domain" not in out

        manifest = tomllib.loads((tmp_path / "pilot-tight.system").read_text())
        run = manifest["run"]
        assert run["dlpno_threshold_preset"] == "TightPNO"
        assert run["dlpno_tcut_pairs"] == ""
        assert run["dlpno_tcut_pno"] == pytest.approx(1e-7)
        assert run["dlpno_tcut_mkn"] == pytest.approx(1e-4)
        assert run["dlpno_residual_domain"] == ""
        assert run["frozen_core_orbitals"] == 0
        assert run["frozen_core_convention"] == "all-electron-explicit"

        records = [
            json.loads(line)
            for line in (tmp_path / "pilot-tight.scf.jsonl").read_text().splitlines()
            if line.strip()
        ]
        done = next(r for r in records if r["event"] == "dlpno_ccsd_done")
        assert done["threshold_preset"] == "TightPNO"
        assert set(done["threshold_applied"]) == {"tcut_pno", "tcut_mkn"}
        assert done["threshold_unsupported"] == ["tcut_pairs"]
        assert done["residual_domain"] == ""
        assert done["n_frozen"] == 0
        assert done["frozen_core_convention"] == "all-electron-explicit"

    def test_sto3g_without_aux_raises(self, tmp_path, monkeypatch):
        # STO-3G ships no default RI aux; DLPNO must refuse rather than guess.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(NotImplementedError, match="aux_basis="):
            run_job(_h2(), basis="sto-3g", method="dlpno-ccsd", output="noaux")

    def test_explicit_aux_basis_override(self, tmp_path, monkeypatch):
        # The aux_basis knob on the DLPNO options unblocks orbital bases with
        # no registered default RI aux (the sto-3g DLPNO batch failures).
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

        res = run_job(
            _h2(),
            basis="sto-3g",
            method="dlpno-ccsd",
            output="stoaux",
            dlpno_ccsd_options=LocalCCSDOptions(aux_basis="cc-pvdz-ri"),
        )
        assert res.converged
        assert res.dlpno_ccsd.converged
        # The explicit aux must be the one actually used (RI line in the log).
        out = (tmp_path / "stoaux.out").read_text()
        assert "(RI: cc-pvdz-ri)" in out

    def test_open_shell_explicit_aux_basis_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions

        res = run_job(
            self._oh(),
            basis="sto-3g",
            method="dlpno-ccsd",
            output="stoauxu",
            dlpno_ccsd_options=DLPNOUCCSDPilotOptions(aux_basis="cc-pvdz-ri"),
        )
        assert res.converged
        assert res.dlpno_ccsd.converged
        out = (tmp_path / "stoauxu.out").read_text()
        assert "RI: cc-pvdz-ri" in out


    def test_out_records_the_resolved_truncation_thresholds(
        self, tmp_path, monkeypatch
    ):
        """The DLPNO-CCSD block must name the thresholds it ran at (#417).

        Companion to the DLPNO-MP2 pin in tests/test_runner_dlpno_mp2.py.
        Both routes are pinned because they resolve their thresholds from
        *different* option classes (`LocalCCSDOptions` vs `DLPNOMP2Options`),
        which is the defect tracked as #448 -- so a single-route pin would
        not prove the row reports the route's own resolved values.
        """
        monkeypatch.chdir(tmp_path)
        run_job(_h2(), basis="cc-pvdz", method="dlpno-ccsd", output="ccthresh")
        out = (tmp_path / "ccthresh.out").read_text()

        assert "Truncation thresholds" in out
        # The operational local solver implements all three coordinates of
        # the shared cross-route NormalPNO policy (#448).
        assert "tcut_pairs 0.0001" in out
        assert "tcut_pno 3.33e-07" in out
        assert "tcut_mkn 0.001" in out


class TestRunJobTNODomainReporting:
    """The public route reports its triples domain, and says when it collapsed.

    `LocalCCSDOptions.tcut_tno` defaults to 0.0, so a default `dlpno-ccsd(t)`
    job truncates nothing. A user who opts into a positive value through
    `dlpno_ccsd_options` used to get no mention of the triples domain anywhere
    in the output: not in the `.out`, not in the structured event, and no
    warning even when a triple had been truncated below the three distinct
    virtuals a triple excitation requires and so contributed exactly zero.
    """

    @staticmethod
    def _h2o():
        a = 1.8897259886
        return Molecule(
            [
                Atom(8, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.793353 * a, -0.613510 * a]),
                Atom(1, [0.0, -0.793353 * a, -0.613510 * a]),
            ],
            charge=0,
            multiplicity=1,
        )

    @staticmethod
    def _options(tcut_tno):
        from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

        return LocalCCSDOptions(
            # This reporting ratchet's 25-triple/19-virtual counts are the
            # pre-#140 all-electron full-space evidence, not a new-default run.
            n_frozen=0,
            tcut_pno=0.0,
            tcut_pairs=0.0,
            tcut_mkn=0.0,
            residual_domain="full",
            compute_triples=True,
            triples_mode="t1",
            tcut_tno=tcut_tno,
        )

    def test_full_domain_reports_domain_and_does_not_warn(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        res = run_job(
            self._h2o(),
            basis="def2-svp",
            method="dlpno-ccsd(t)",
            dlpno_ccsd_options=self._options(0.0),
            output="full",
        )
        assert res.converged
        out = (tmp_path / "full.out").read_text()
        assert "triples / avg TNOs" in out
        # def2-SVP water: 24 basis functions, 5 occupied -> 19 virtuals, and
        # the full domain is the whole virtual space for every triple.
        assert "(smallest domain: 19)" in out
        assert "Residual domain" in out and "= full" in out
        assert "truncated" not in out
        assert res.dlpno_ccsd.n_degenerate_tno_triples == 0

    def test_collapsed_triples_warn_in_the_out(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = run_job(
            self._h2o(),
            basis="def2-svp",
            method="dlpno-ccsd(t)",
            dlpno_ccsd_options=self._options(1e-5),
            output="coarse",
        )
        assert res.converged
        assert res.dlpno_ccsd.n_degenerate_tno_triples >= 1

        out = (tmp_path / "coarse.out").read_text()
        assert "(smallest domain: 1)" in out
        assert "WARNING" in out
        assert "below three virtuals" in out
        assert "contribute exactly zero" in out

    def test_structured_event_carries_the_domain(self, tmp_path, monkeypatch):
        import json

        monkeypatch.chdir(tmp_path)
        run_job(
            self._h2o(),
            basis="def2-svp",
            method="dlpno-ccsd(t)",
            dlpno_ccsd_options=self._options(1e-5),
            output="ev",
            structured_log=True,
        )
        records = [
            json.loads(line)
            for line in (tmp_path / "ev.scf.jsonl").read_text().splitlines()
            if line.strip()
        ]
        done = [r for r in records if r.get("event") == "dlpno_ccsd_done"]
        assert len(done) == 1
        assert done[0]["threshold_settings"]["tcut_tno"] == pytest.approx(
            1e-5
        )
        assert done[0]["residual_domain"] == "full"
        assert done[0]["n_tno_triples"] == 25
        assert done[0]["min_tno_per_triple"] == 1
        assert done[0]["n_degenerate_tno_triples"] >= 1
        assert done[0]["avg_tno_per_triple"] < 19.0

        manifest = tomllib.loads((tmp_path / "ev.system").read_text())
        manifest_run = manifest["run"]
        assert manifest_run["dlpno_tcut_tno"] == pytest.approx(1e-5)
        assert manifest_run["dlpno_tcut_pno_singles"] == ""
        assert manifest_run["dlpno_residual_domain"] == "full"

    @pytest.mark.parametrize(
        "mode,algorithm",
        [
            ("local", "DLPNO-(T0), diagonal local occupied-Fock"),
            ("t1", "rotated-occupied (T1) compatibility correction"),
            ("exact", "canonical (T) contraction on local amplitudes"),
        ],
    )
    def test_closed_shell_triples_mode_is_bound_to_every_artifact(
        self, mode, algorithm, tmp_path, monkeypatch
    ):
        from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions

        monkeypatch.chdir(tmp_path)
        run_job(
            _h2(),
            basis="def2-svp",
            method="dlpno-ccsd(t)",
            dlpno_ccsd_options=LocalCCSDOptions(triples_mode=mode),
            output=f"mode-{mode}",
            structured_log=True,
        )

        out = (tmp_path / f"mode-{mode}.out").read_text()
        assert f"{algorithm} (triples_mode={mode!r})" in out

        manifest = tomllib.loads(
            (tmp_path / f"mode-{mode}.system").read_text()
        )["run"]
        assert manifest["dlpno_triples_mode"] == mode
        assert manifest["dlpno_triples_algorithm"] == algorithm
        if mode == "exact":
            assert manifest["dlpno_tcut_tno"] == ""

        records = [
            json.loads(line)
            for line in (
                tmp_path / f"mode-{mode}.scf.jsonl"
            ).read_text().splitlines()
            if line.strip()
        ]
        done = next(r for r in records if r["event"] == "dlpno_ccsd_done")
        assert done["triples_mode"] == mode
        assert done["triples_algorithm"] == algorithm

    def test_pilot_route_has_no_tno_fields_and_still_runs(
        self, tmp_path, monkeypatch
    ):
        """The open-shell pilot result carries no TNO map; reporting must not
        assume one exists. The pilot is opt-in via DLPNOUCCSDPilotOptions."""
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions
        oh = Molecule(
            [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.832])],
            charge=0,
            multiplicity=2,
        )
        res = run_job(
            oh, basis="def2-svp", method="dlpno-ccsd(t)", output="pilot",
            dlpno_ccsd_options=DLPNOUCCSDPilotOptions(),
            structured_log=True,
        )
        assert res.converged
        out = (tmp_path / "pilot.out").read_text()
        # The pilot builds no TNO domain, so the domain line is omitted
        # rather than printed as zeros.
        assert "triples / avg TNOs" not in out
        assert "NormalPNO (supported subset)" in out
        assert "tcut_pno 3.33e-07" in out
        assert "tcut_pairs 0.0001" not in out
        assert "tcut_mkn 0.001" not in out
        assert "Unsupported cutoffs" in out
        assert "tcut_pairs, tcut_mkn" in out
        assert "Frozen core orbitals = 1" in out
        assert "published count-only default" in out

        manifest = tomllib.loads((tmp_path / "pilot.system").read_text())
        run = manifest["run"]
        assert run["dlpno_threshold_preset"] == "NormalPNO"
        assert run["dlpno_tcut_pairs"] == ""
        assert run["dlpno_tcut_pno"] == pytest.approx(3.33e-7)
        assert run["dlpno_tcut_mkn"] == ""
        assert run["dlpno_triples_mode"] == "pilot"
        assert run["dlpno_triples_algorithm"] == "dense correctness-pilot (T)"
        assert run["frozen_core_orbitals"] == 1
        assert (
            run["frozen_core_convention"]
            == "orca-6.1-table-2.69-count-only-default"
        )

        records = [
            json.loads(line)
            for line in (tmp_path / "pilot.scf.jsonl").read_text().splitlines()
            if line.strip()
        ]
        done = next(r for r in records if r["event"] == "dlpno_uccsd_done")
        assert set(done["threshold_applied"]) == {"tcut_pno"}
        assert done["threshold_unsupported"] == ["tcut_pairs", "tcut_mkn"]
        assert done["triples_mode"] == "pilot"
        assert done["triples_algorithm"] == "dense correctness-pilot (T)"
        assert done["n_frozen"] == 1
        assert (
            done["frozen_core_convention"]
            == "orca-6.1-table-2.69-count-only-default"
        )
