"""run_job integration for method="dlpno-mp2" (M5a wiring)."""

from __future__ import annotations

import json
import tomllib

import numpy as np
import pytest
from vibeqc import run_job
from vibeqc._vibeqc_core import Atom, Molecule, UHFOptions
from vibeqc.runner import _default_open_shell_dlpno_uhf_options

ANGSTROM_TO_BOHR = 1.8897259886


def _h2o():
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
            Atom(1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=1,
    )


def _h2():
    return Molecule(
        [
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.4]),
        ],
        charge=0,
        multiplicity=1,
    )


class TestRunJobDLPNOMP2:
    def test_open_shell_default_uhf_cap_allows_dlpno_reference_tail(self):
        oh = Molecule(
            [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.832])],
            charge=0,
            multiplicity=2,
        )

        opts = _default_open_shell_dlpno_uhf_options("dlpno-mp2", oh, None)

        assert opts is not None
        assert opts.max_iter >= 250

        explicit = UHFOptions()
        explicit.max_iter = 37
        assert (
            _default_open_shell_dlpno_uhf_options("dlpno-mp2", oh, explicit)
            is explicit
        )

    def test_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        res = run_job(_h2o(), basis="def2-svp", method="dlpno-mp2", output="h2o")

        # Result proxy: SCF attributes forward, DLPNO result attached.
        assert res.converged
        assert hasattr(res, "dlpno_mp2")
        r = res.dlpno_mp2
        assert r.converged
        assert res.energy_total == pytest.approx(r.e_total, abs=1e-12)
        # NormalPNO plus the published one-orbital frozen core is the public
        # default for H2O/def2-SVP.  Pin that resolved recipe, rather than the
        # historical all-electron threshold mix that preceded #140/#448.
        #
        # Re-pinned by #65: -0.2040867439 -> -0.2041762797, a shift of
        # -0.0895 mHa. The default pair density is now Riplinger and Neese
        # Eq. 23, which retains more PNOs at the same tcut_pno, so the
        # truncated space recovers more correlation. This pin tracks the
        # public default deliberately, so it moves when the default moves --
        # unlike the retained rows in test_release_paper_dlpno_m16.py, which
        # pin their convention explicitly to stay fixed.
        assert r.n_frozen == 1
        assert r.e_corr == pytest.approx(-0.2041762797, abs=5e-7)

        out = (tmp_path / "h2o.out").read_text()
        assert "DLPNO-MP2 (Pinski 2015; RI: def2-svp-rifit)" in out
        assert "E(DLPNO-MP2 total)" in out
        assert "E(PNO truncation corr)" in out

        # Citation surface: the method papers must be reachable from the
        # job's reference output (citation discipline, CLAUDE.md § 8).
        refs = (tmp_path / "h2o.references").read_text()
        assert "Pinski" in refs
        assert "Foster" in refs or "Boys" in refs
        bib = (tmp_path / "h2o.bibtex").read_text()
        assert "pinski_dlpno_mp2_2015" in bib

    def test_parity_with_canonical_mp2_runner(self, tmp_path, monkeypatch):
        """DLPNO-MP2 through run_job tracks canonical MP2 through run_job."""
        monkeypatch.chdir(tmp_path)
        res_dlpno = run_job(
            _h2o(), basis="def2-svp", method="dlpno-mp2", output="a"
        )
        res_mp2 = run_job(_h2o(), basis="def2-svp", method="mp2", output="b")
        # Conventional (non-DF) MP2 vs DF-based DLPNO-MP2: difference is
        # RI-fit error + truncation — well under 1 mHa on this system.
        assert abs(
            res_dlpno.dlpno_mp2.e_corr - res_mp2.mp2.e_correlation
        ) < 1e-3

    def test_local_df_run_job_avoids_global_density_fitting(
        self, tmp_path, monkeypatch
    ):
        """local_df=True builds local sub-basis integrals from the public route."""
        from vibeqc.dlpno.mp2 import DLPNOMP2Options
        import vibeqc.density_fitting as dfmod

        class ForbiddenDensityFitting:
            def __init__(self, *args, **kwargs):
                raise AssertionError("global DensityFitting constructor was used")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(dfmod, "DensityFitting", ForbiddenDensityFitting)

        opts = DLPNOMP2Options(
            localise="none",
            local_df=True,
            fit_buffer=1.0e9,
            tcut_pno=0.0,
            tcut_pno_weak=0.0,
            tcut_mkn=0.0,
            tcut_pairs=0.0,
            tcut_pairs_weak=0.0,
        )
        res = run_job(
            _h2(),
            basis="def2-svp",
            method="dlpno-mp2",
            output="h2-local",
            dlpno_options=opts,
        )

        assert res.converged
        assert res.dlpno_mp2.converged
        assert res.energy_total == pytest.approx(res.dlpno_mp2.e_total, abs=1e-12)

    def test_open_shell_routes_to_ump2(self, tmp_path, monkeypatch):
        # method='dlpno-mp2' on an open-shell reference auto-routes to the
        # UHF-based DLPNO-UMP2 path (M1-M1c + screening).
        monkeypatch.chdir(tmp_path)
        oh = Molecule(
            [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.832])],
            charge=0,
            multiplicity=2,
        )
        res = run_job(oh, basis="def2-svp", method="dlpno-mp2", output="oh")
        assert res.converged
        assert hasattr(res, "dlpno_ump2")
        r = res.dlpno_ump2
        assert res.energy_total == pytest.approx(r.e_total, abs=1e-12)
        # spin-channel decomposition is present and physical
        assert r.e_aa < 0.0 and r.e_bb < 0.0 and r.e_ab < -1e-2
        assert r.e_corr == pytest.approx(r.e_aa + r.e_bb + r.e_ab, abs=1e-12)

        out = (tmp_path / "oh.out").read_text()
        assert "DLPNO-UMP2 (UHF reference; RI:" in out
        assert "Saitow" not in out
        assert "E(aa same-spin)" in out
        assert "E(DLPNO-UMP2 total)" in out

        # Saitow 2017 is an open-shell CC paper whose central single-spatial-
        # orbital construction is not this independent-spin UMP2 method.
        refs = (tmp_path / "oh.references").read_text()
        assert "Pinski" in refs
        assert "Saitow" not in refs
        bib = (tmp_path / "oh.bibtex").read_text()
        assert "pinski_dlpno_mp2_2015" in bib
        assert "saitow_openshell_dlpno_2017" not in bib

    def test_open_shell_options_passed_through(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.ump2 import DLPNOUMP2Options

        oh = Molecule(
            [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.832])],
            charge=0,
            multiplicity=2,
        )
        res = run_job(
            oh,
            basis="def2-svp",
            method="dlpno-mp2",
            output="ohb",
            dlpno_options=DLPNOUMP2Options(localise="boys", tcut_pno=0.0, n_frozen=1),
        )
        r = res.dlpno_ump2
        assert r.localise == "boys" and r.n_frozen == 1

    def test_frozen_core_via_options(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.mp2 import DLPNOMP2Options

        res = run_job(
            _h2o(),
            basis="def2-svp",
            method="dlpno-mp2",
            output="fc",
            dlpno_options=DLPNOMP2Options(n_frozen=1),
        )
        r = res.dlpno_mp2
        assert r.n_frozen == 1
        assert all(i >= 1 and j >= 1 for (i, j) in r.pair_energies)
        out = (tmp_path / "fc.out").read_text()
        assert "Frozen core orbitals = 1 (explicit orbital count)" in out

    def test_aux_basis_kwarg_reaches_route(self, tmp_path, monkeypatch):
        """run_job(aux_basis=...) selects the DLPNO-MP2 RI basis (BUG 113).

        Before the fix the kwarg was wired to the SCF options only and the
        route silently auto-resolved its own RI basis, so an explicit user
        request was dropped.
        """
        monkeypatch.chdir(tmp_path)
        res = run_job(
            _h2o(),
            basis="def2-svp",
            method="dlpno-mp2",
            output="auxkw",
            aux_basis="cc-pvdz-ri",
        )
        assert res.dlpno_mp2.converged
        out = (tmp_path / "auxkw.out").read_text()
        assert "DLPNO-MP2 (Pinski 2015; RI: cc-pvdz-ri)" in out
        # The auto-resolved default must NOT appear -- the kwarg wins.
        assert "RI: def2-svp-rifit" not in out

    def test_aux_basis_kwarg_conflict_raises(self, tmp_path, monkeypatch):
        """An options-level aux_basis that contradicts the kwarg raises."""
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.mp2 import DLPNOMP2Options

        opts = DLPNOMP2Options(aux_basis="cc-pvtz-ri")
        stem = tmp_path / "conflict"
        with pytest.raises(ValueError, match="contradicts"):
            run_job(
                _h2o(),
                basis="def2-svp",
                method="dlpno-mp2",
                output=stem,
                aux_basis="cc-pvdz-ri",
                dlpno_options=opts,
            )
        assert not stem.with_suffix(".out").exists()

    def test_aux_basis_unresolvable_fails_loudly(self, tmp_path, monkeypatch):
        """An unresolvable explicit aux basis raises, not silently ignored."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="no shells loaded"):
            run_job(
                _h2(),
                basis="def2-svp",
                method="dlpno-mp2",
                output="badaux",
                aux_basis="no-such-basis-xyz",
            )

    def test_out_records_the_resolved_truncation_thresholds(
        self, tmp_path, monkeypatch
    ):
        """A DLPNO run's own .out must name the thresholds it ran at.

        Issue #417: `tcut_pno` / `tcut_pairs` / `tcut_mkn` appeared in NO
        artifact -- not the .out, not the .system -- so a paper claiming a
        named truncation level was unprovable from the run that produced it.
        The block already reported pairs kept/screened, average PNOs per pair
        and the frozen-core count; it simply never reported the cutoffs that
        produced them.

        Pinned against explicitly-set, non-default values so the row is proven
        to carry the *resolved* thresholds rather than a hardcoded string.
        """
        monkeypatch.chdir(tmp_path)
        from vibeqc.dlpno.mp2 import DLPNOMP2Options

        exact_text_cutoff = 3.141592653589793e-9
        opts = DLPNOMP2Options(
            tcut_pno=exact_text_cutoff,
            tcut_pairs=1e-7,
            tcut_mkn=1e-4,
        )
        run_job(
            _h2o(),
            basis="def2-svp",
            method="dlpno-mp2",
            output="thresh",
            dlpno_options=opts,
            structured_log=True,
        )
        out = (tmp_path / "thresh.out").read_text()

        assert "Truncation thresholds" in out
        # Publications order the knobs TCutPairs, TCutPNO, TCutMKN
        # (Liakos 2015 Table 1); the row follows that order.
        assert "tcut_pairs 1e-07" in out
        assert f"tcut_pno {exact_text_cutoff!r}" in out
        assert "tcut_mkn 0.0001" in out

        manifest = tomllib.loads((tmp_path / "thresh.system").read_text())
        manifest_run = manifest["run"]
        assert manifest_run["dlpno_tcut_pairs"] == pytest.approx(1e-7)
        assert manifest_run["dlpno_tcut_pno"] == pytest.approx(exact_text_cutoff)
        assert manifest_run["dlpno_tcut_mkn"] == pytest.approx(1e-4)
        assert manifest_run["dlpno_tcut_pairs_weak"] == pytest.approx(1e-4)
        assert manifest_run["dlpno_tcut_pno_weak"] == pytest.approx(3.33e-6)
        assert manifest_run["dlpno_tcut_pno_singles"] == ""
        assert manifest_run["dlpno_tcut_tno"] == ""

        records = [
            json.loads(line)
            for line in (tmp_path / "thresh.scf.jsonl").read_text().splitlines()
            if line.strip()
        ]
        done = next(r for r in records if r["event"] == "dlpno_mp2_done")
        assert done["threshold_settings"] == {
            "tcut_pairs": pytest.approx(1e-7),
            "tcut_pno": pytest.approx(exact_text_cutoff),
            "tcut_mkn": pytest.approx(1e-4),
            "tcut_pairs_weak": pytest.approx(1e-4),
            "tcut_pno_weak": pytest.approx(3.33e-6),
        }
