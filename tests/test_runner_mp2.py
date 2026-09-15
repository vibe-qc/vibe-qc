"""End-to-end ``run_job`` exposure of post-SCF MP2 (+ SCS / SOS).

vibe-qc has had a native C++ MP2 (with ``c_os`` / ``c_ss`` spin-component
scaling) for a while, but it was not reachable through the public
``run_job`` dispatcher — only via the low-level ``run_mp2`` /
``run_scs_mp2`` runners.  These tests pin the dispatcher wiring:

  * ``method="mp2"`` runs the RHF SCF then the MP2 correction and reports
    ``E(MP2 total) = E(HF) + E(corr)``;
  * ``method="scs-mp2"`` / ``"sos-mp2"`` apply Grimme's / Jung's spin
    scaling (factors sourced from :data:`vibeqc.correlation._MP2_SCALES`);
  * the ``.out`` carries an MP2 block and the header names the method;
  * the ``.references`` sibling fires the right §8 citation per variant;
  * open-shell MP2 (UMP2) fails fast with a clear message (not silently
    running RHF on a radical).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from vibeqc import InsufficientMemoryError, d3bj_params_for, run_job
from vibeqc.correlation import _MP2_SCALES
from vibeqc.molecule import Atom, Molecule

_A2B = 1.8897259886


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.7572 * _A2B, 0.5865 * _A2B]),
            Atom(1, [0.0, -0.7572 * _A2B, 0.5865 * _A2B]),
        ],
        0,
        1,
    )


def test_run_job_mp2_total_energy(tmp_path: Path) -> None:
    """method="mp2" reproduces the native run_mp2 e_total = e_hf + e_corr."""
    from vibeqc._vibeqc_core import (BasisSet, MP2Options, RHFOptions, run_mp2,
                                     run_rhf)

    mol = _h2o()
    r = run_job(mol, basis="sto-3g", method="mp2",
                output=str(tmp_path / "h2o"), verbose=0)
    mp2 = r.mp2
    # Independent reference: native RHF + native MP2.
    bas = BasisSet(mol, "sto-3g")
    rhf = run_rhf(mol, bas, RHFOptions())
    ref = run_mp2(mol, bas, rhf, MP2Options())
    assert mp2.e_correlation == pytest.approx(ref.e_correlation, abs=1e-10)
    assert mp2.e_total == pytest.approx(ref.e_total, abs=1e-10)
    assert mp2.e_total == pytest.approx(mp2.e_hf + mp2.e_correlation, abs=1e-12)


def test_run_job_mp2_terminal_outputs_use_method_total(tmp_path: Path) -> None:
    """Terminal machine records expose MP2 total while retaining the SCF part."""
    stem = tmp_path / "h2o_terminal"
    result = run_job(
        _h2o(),
        basis="sto-3g",
        method="mp2",
        output=stem,
        structured_log=True,
        output_qvf=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    assert result.energy_total == pytest.approx(result.mp2.e_total, abs=1e-12)
    assert abs(float(result.energy_total) - float(result.energy)) > 1e-4

    events = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    mp2_done = next(event for event in events if event["event"] == "mp2_done")
    job_end = next(event for event in events if event["event"] == "job_end")
    assert mp2_done["e_total"] == pytest.approx(result.energy_total, abs=1e-12)
    assert job_end["energy"] == pytest.approx(result.energy_total, abs=1e-12)
    assert job_end["e_scf"] == pytest.approx(result.energy, abs=1e-12)

    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    assert manifest["progress"]["phase"] == "solver"
    assert manifest["progress"]["energy_eh"] == pytest.approx(
        result.energy_total, abs=1e-12
    )


def test_run_job_mp2_terminal_total_composes_dispersion_once(tmp_path: Path) -> None:
    """A dispersion wrapper adds to the MP2 total, not the SCF component."""
    stem = tmp_path / "h2o_mp2_d3"
    result = run_job(
        _h2o(),
        basis="sto-3g",
        method="mp2",
        dispersion=d3bj_params_for("pbe"),
        output=stem,
        structured_log=True,
        output_qvf=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    expected = float(result.mp2.e_total) + float(result.e_dispersion)
    assert result.energy_total == pytest.approx(expected, abs=1e-12)

    events = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    job_end = next(event for event in events if event["event"] == "job_end")
    assert job_end["energy"] == pytest.approx(expected, abs=1e-12)
    assert job_end["e_scf"] == pytest.approx(result.energy, abs=1e-12)
    assert job_end["e_dispersion"] == pytest.approx(
        result.e_dispersion, abs=1e-12
    )
    assert job_end["e_total"] == pytest.approx(expected, abs=1e-12)

    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    assert manifest["progress"]["energy_eh"] == pytest.approx(expected, abs=1e-12)


def test_run_job_mp2_default_output_names_canonical_route(tmp_path: Path) -> None:
    """run_job(method="mp2") defaults to in-core canonical MP2, not RI."""
    stem = tmp_path / "h2o"
    result = run_job(_h2o(), basis="sto-3g", method="mp2",
                     output=str(stem), verbose=0)

    out = stem.with_suffix(".out").read_text()
    assert "Algorithm              = in-core conventional canonical MP2" in out
    assert "memory=incore" in out
    assert "no RI auxiliary basis" in out
    assert (
        "Frozen core orbitals = 1 (ORCA 6.1 published count-only default)"
        in out
    )
    assert result.mp2.ri_residual_reported is False
    assert result.mp2.e_os_ri_residual == 0.0
    assert result.mp2.e_ss_ri_residual == 0.0


def test_run_job_mp2_options_enable_ri_route(tmp_path: Path) -> None:
    """mp2_options reaches the native RMP2 kernel and can request RI-MP2."""
    from vibeqc._vibeqc_core import BasisSet, MP2Options, RHFOptions, run_mp2
    from vibeqc._vibeqc_core import run_rhf

    mol = _h2o()
    stem = tmp_path / "h2o_ri"
    opts = MP2Options()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-rifit"

    result = run_job(
        mol,
        basis="def2-svp",
        method="mp2",
        mp2_options=opts,
        output=str(stem),
        verbose=0,
    )

    bas = BasisSet(mol, "def2-svp")
    rhf = run_rhf(mol, bas, RHFOptions())
    ref = run_mp2(mol, bas, rhf, opts)
    assert result.mp2.e_total == pytest.approx(ref.e_total, abs=1e-9)

    out = stem.with_suffix(".out").read_text()
    assert "Algorithm              = RI-MP2" in out
    assert "aux_basis=def2-svp-rifit" in out
    refs = stem.with_suffix(".references").read_text()
    assert "feyereisen_rimp2_1993" in refs


def test_run_job_direct_mp2_process_budget_reserves_headroom(tmp_path: Path) -> None:
    """A process cap is converted to a smaller method-owned native cap."""
    from vibeqc import MP2Options

    stem = tmp_path / "h2o_direct_budget"
    opts = MP2Options()
    opts.memory_mode = "direct"
    opts.requested_memory_bytes = 16 * 1024**2
    process_budget = 256 * 1024**2

    result = run_job(
        _h2o(),
        basis="sto-3g",
        method="mp2",
        mp2_options=opts,
        memory_budget_bytes=process_budget,
        output=stem,
        structured_log=True,
        verbose=0,
    )

    assert result.mp2.memory_mode_used == "direct"
    events = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    memory = next(event for event in events if event["event"] == "memory_estimate")
    native_budget = memory["native_workspace_budget_bytes"]
    assert native_budget == opts.requested_memory_bytes
    assert result.mp2.workspace_bytes <= native_budget
    assert "10.1016/0009-2614(88)85250-3" in (
        stem.with_suffix(".references").read_text()
    )


@pytest.mark.parametrize("process_budget", [None, 256 * 1024**2])
def test_run_job_rejects_impossible_method_cap_before_scf(
    tmp_path: Path,
    monkeypatch,
    process_budget,
) -> None:
    """A tiny option cap fails before SCF with known or unknown process RAM."""
    from vibeqc import MP2Options

    opts = MP2Options()
    opts.memory_mode = "direct"
    opts.requested_memory_bytes = 1

    def fail_if_scf_runs(*args, **kwargs):
        raise AssertionError("SCF ran before the impossible MP2 cap was rejected")

    monkeypatch.setattr("vibeqc.runner.available_memory_bytes", lambda: 0)
    monkeypatch.setattr("vibeqc.runner.run_rhf", fail_if_scf_runs)
    with pytest.raises(InsufficientMemoryError, match="minimum modeled MP2"):
        run_job(
            _h2o(),
            basis="sto-3g",
            method="mp2",
            mp2_options=opts,
            memory_budget_bytes=process_budget,
            output=tmp_path / "impossible_cap",
            verbose=0,
        )


def test_run_job_df_direct_mp2_fires_runtime_algorithm_citation(
    tmp_path: Path,
) -> None:
    from vibeqc import MP2Options

    stem = tmp_path / "df_direct"
    opts = MP2Options()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-rifit"
    opts.memory_mode = "direct"
    opts.requested_memory_bytes = 64 * 1024**2

    result = run_job(
        _h2o(),
        basis="def2-svp",
        method="mp2",
        mp2_options=opts,
        memory_budget_bytes=256 * 1024**2,
        output=stem,
        verbose=0,
    )

    assert result.mp2.memory_mode_used == "direct"
    refs = stem.with_suffix(".references").read_text()
    assert "10.1021/acs.jctc.2c00640" in refs
    assert "10.1016/0009-2614(93)87156-W" in refs


def test_run_job_disk_mp2_fires_semidirect_citation_and_cleans_scratch(
    tmp_path: Path,
) -> None:
    from vibeqc import MP2Options

    stem = tmp_path / "disk"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    opts = MP2Options()
    opts.memory_mode = "disk"
    opts.requested_memory_bytes = 16 * 1024**2
    opts.scratch_directory = str(scratch)

    result = run_job(
        _h2o(),
        basis="sto-3g",
        method="mp2",
        mp2_options=opts,
        memory_budget_bytes=256 * 1024**2,
        output=stem,
        verbose=0,
    )

    assert result.mp2.memory_mode_used == "disk"
    assert result.mp2.disk_bytes > 0
    assert list(scratch.iterdir()) == []
    refs = stem.with_suffix(".references").read_text()
    assert "10.1016/0009-2614(88)85250-3" in refs
    assert "10.1016/0009-2614(90)80030-H" in refs


@pytest.mark.parametrize("variant", ["scs-mp2", "sos-mp2"])
def test_run_job_spin_component_scaling(tmp_path: Path, variant: str) -> None:
    """SCS / SOS scale the unscaled OS/SS components by _MP2_SCALES."""
    mol = _h2o()
    plain = run_job(mol, basis="sto-3g", method="mp2",
                    output=str(tmp_path / "plain"), verbose=0).mp2
    scaled = run_job(mol, basis="sto-3g", method=variant,
                     output=str(tmp_path / variant), verbose=0).mp2
    c_os, c_ss = _MP2_SCALES[variant]
    # Unscaled components are identical across variants.
    assert scaled.e_os == pytest.approx(plain.e_os, abs=1e-12)
    assert scaled.e_ss == pytest.approx(plain.e_ss, abs=1e-12)
    # The correlation energy is the scaled sum.
    assert scaled.e_correlation == pytest.approx(
        c_os * plain.e_os + c_ss * plain.e_ss, abs=1e-12
    )
    if variant == "sos-mp2":
        assert c_ss == 0.0  # SOS drops same-spin entirely


def test_run_job_mp2_writes_block_and_header(tmp_path: Path) -> None:
    """The .out names the method in the header and carries an MP2 block."""
    run_job(_h2o(), basis="sto-3g", method="scs-mp2",
            output=str(tmp_path / "h2o"), verbose=0)
    out = (tmp_path / "h2o.out").read_text()
    assert "RHF + SCS-MP2" in out          # job header
    assert "Moller" in out                 # MP2 block heading
    assert "E(MP2 correlation)" in out
    assert "E(SCS-MP2 total)" in out


@pytest.mark.parametrize("variant,needle", [
    ("mp2", "10.1103/PhysRev.46.618"),       # Moller-Plesset 1934
    ("scs-mp2", "10.1063/1.1569242"),        # Grimme SCS 2003
    ("sos-mp2", "10.1063/1.1809602"),        # Jung SOS 2004
])
def test_run_job_mp2_fires_citation(tmp_path: Path, variant: str,
                                    needle: str) -> None:
    """Each variant fires its §8 citation into the .references sibling.

    Routing keys off the *original* method (not the resolved RHF SCF), so
    the correlation papers reach the user-facing reference block."""
    run_job(_h2o(), basis="sto-3g", method=variant,
            output=str(tmp_path / "h2o"), verbose=0)
    refs = (tmp_path / "h2o.references").read_text()
    assert needle in refs
    # MP2 lineage paper is always present.
    assert "PhysRev.46.618" in refs


def test_run_job_open_shell_mp2_runs_ump2(tmp_path: Path) -> None:
    """Open-shell MP2 runs the native UMP2 (UHF reference) and matches run_ump2;
    the header names the UHF reference and the block is labelled UMP2."""
    from vibeqc._vibeqc_core import (Atom as CAtom, BasisSet, Molecule as CMol,
                                     UHFOptions, UMP2Options, run_uhf, run_ump2)
    o2 = Molecule([Atom(8, [0, 0, 0]), Atom(8, [0, 0, 1.2 * _A2B])], 0, 3)
    r = run_job(o2, basis="sto-3g", method="mp2",
                output=str(tmp_path / "o2"), verbose=0)
    cm = CMol([CAtom(8, [0, 0, 0]), CAtom(8, [0, 0, 1.2 * _A2B])], 0, 3)
    bas = BasisSet(cm, "sto-3g")
    uhf = run_uhf(cm, bas, UHFOptions())
    ump = run_ump2(cm, bas, uhf, UMP2Options())
    assert r.mp2.e_total == pytest.approx(ump.e_total, abs=1e-9)
    assert r.mp2.e_correlation == pytest.approx(ump.e_correlation, abs=1e-9)
    out = (tmp_path / "o2.out").read_text()
    assert "UHF + MP2" in out
    assert "Moller-Plesset MP2 (UMP2)" in out


def test_run_job_mp2_reference_rohf(tmp_path: Path) -> None:
    """method="mp2" + mp2_reference="rohf" runs ROHF + semicanonical ROHF-MP2
    (not UMP2): the block names the ROHF reference and reports the Knowles
    singles term, the energy matches the standalone run_rohf_mp2, and the
    rohf-mp2 citation (Knowles 1991) fires."""
    from vibeqc import BasisSet, run_rohf, run_rohf_mp2

    oh = Molecule([Atom(8, [0, 0, 0]), Atom(1, [0, 0, 0.97 * _A2B])], 0, 2)
    stem = tmp_path / "oh"
    r = run_job(oh, basis="cc-pvdz", method="mp2", mp2_reference="rohf",
                aux_basis="def2-svp-rifit", output=str(stem),
                structured_log=True, verbose=0)
    # Independent reference: both public routes use the shared published
    # count-only frozen-core default (one frozen spatial orbital for OH).
    bas = BasisSet(oh, "cc-pvdz")
    ref = run_rohf_mp2(
        oh,
        bas,
        run_rohf(oh, bas),
        aux_basis="def2-svp-rifit",
    )
    assert r.mp2.e_corr == pytest.approx(ref.e_corr, abs=1e-9)
    assert abs(r.mp2.e_singles) > 1e-4          # ROHF Brillouin singles present
    out = (stem.with_suffix(".out")).read_text()
    assert "Moller-Plesset MP2 (ROHF-MP2)" in out
    assert "E(ROHF reference)" in out
    assert "singles, Brillouin" in out
    assert "RI semicanonical ROHF-MP2" in out
    assert "aux_basis=def2-svp-rifit" in out
    events = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    mp2_done = next(event for event in events if event["event"] == "mp2_done")
    assert mp2_done["density_fit"] is True
    assert mp2_done["aux_basis"] == "def2-svp-rifit"
    refs = (stem.with_suffix(".references")).read_text()
    assert "knowles_rmp2_1991" in refs or "Knowles" in refs


def test_run_job_rohf_mp2_enforces_process_budget_before_scf(tmp_path: Path) -> None:
    """The dense Python oracle is process-admitted, not native-slab planned."""
    from vibeqc import InsufficientMemoryError

    oh = Molecule([Atom(8, [0, 0, 0]), Atom(1, [0, 0, 0.97 * _A2B])], 0, 2)
    with pytest.raises(InsufficientMemoryError, match="ABORTING"):
        run_job(
            oh,
            basis="cc-pvdz",
            method="mp2",
            mp2_reference="rohf",
            memory_budget_bytes=1,
            output=str(tmp_path / "oh_budget"),
            verbose=0,
        )


@pytest.mark.parametrize("variant", ["scs-mp2", "sos-mp2"])
def test_run_job_open_shell_spin_scaling(tmp_path: Path, variant: str) -> None:
    """Open-shell SCS/SOS scale the native UMP2 spin channels (αβ vs αα+ββ)."""
    from vibeqc._vibeqc_core import (Atom as CAtom, BasisSet, Molecule as CMol,
                                     UHFOptions, UMP2Options, run_uhf, run_ump2)
    o2 = Molecule([Atom(8, [0, 0, 0]), Atom(8, [0, 0, 1.2 * _A2B])], 0, 3)
    r = run_job(o2, basis="sto-3g", method=variant,
                output=str(tmp_path / variant), verbose=0)
    c_os, c_ss = _MP2_SCALES[variant]
    cm = CMol([CAtom(8, [0, 0, 0]), CAtom(8, [0, 0, 1.2 * _A2B])], 0, 3)
    bas = BasisSet(cm, "sto-3g")
    uhf = run_uhf(cm, bas, UHFOptions())
    opts = UMP2Options()
    opts.c_os, opts.c_ss = c_os, c_ss
    ump = run_ump2(cm, bas, uhf, opts)
    assert r.mp2.e_total == pytest.approx(ump.e_total, abs=1e-9)
