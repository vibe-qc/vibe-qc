"""Unit tests for the ORCA output parser.

The parser feeds reference numbers into the ORCA parity matrix
(``tests/test_parity_vs_orca.py``) — a silent misparse would poison
every cell. These tests pin it against compact ORCA-6.1-style samples
and prove the mandatory self-check fires on a corrupted file.

No ORCA / vq / vibe-qc needed; the samples are inline so the repo does
not need to carry full generated ORCA logs as documentation artefacts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from examples.regression.parity_matrix_orca.parse_orca import (
    OrcaParseError,
    parse_orca_mp2_output,
    parse_orca_output,
)

_HF_SAMPLE = """
Program Version 6.1.1  -  RELEASE  -

----------------------------
TOTAL SCF ENERGY
----------------------------
Nuclear Repulsion  :             9.0882937241424 Eh
One Electron Energy:          -122.87357160719932 Eh
Two Electron Energy:            37.77685108022314 Eh
Total Energy       :           -76.00842680283379 Eh

ORBITAL ENERGIES
  NO   OCC          E(Eh)            E(eV)
   0   2.0000     -20.557409      -559.3955
   1   2.0000      -1.343201       -36.5502
   2   2.0000      -0.707331       -19.2471
   3   2.0000      -0.559101       -15.2130
   4   2.0000      -0.498315       -13.5590
   5   0.0000       0.205300         5.5865

FINAL SINGLE POINT ENERGY       -76.008426802834
"""

_DFT_SAMPLE = """
Program Version 6.1.1  -  RELEASE  -

----------------------------
TOTAL SCF ENERGY
----------------------------
Nuclear Repulsion  :             9.0882937241424 Eh
One Electron Energy:          -122.91280255103666 Eh
Two Electron Energy:            37.50407944224192 Eh
Total Energy       :           -76.32042938465234 Eh

DFT components
E(XC)             :             -9.259978237268 Eh

ORBITAL ENERGIES
  NO   OCC          E(Eh)            E(eV)
   0   2.0000     -18.881700      -513.7954
   1   2.0000      -0.972314       -26.4580
   2   2.0000      -0.493202       -13.4208
   3   2.0000      -0.336112        -9.1463
   4   2.0000      -0.223708        -6.0876
   5   0.0000       0.033149         0.9014

FINAL SINGLE POINT ENERGY       -76.320429384652
"""


def test_parse_hf_sample() -> None:
    """The HF/6-31G* H2O sample parses into the expected decomposition."""
    out = parse_orca_output(_HF_SAMPLE)

    assert out["code"] == "orca"
    assert out["code_version"] == "6.1.1"
    # HF has no XC term.
    assert out["e_xc"] == 0.0
    # Values straight out of the sample's TOTAL SCF ENERGY block.
    assert out["e_nuc"] == pytest.approx(9.0882937241424, abs=1e-10)
    assert out["e_1e"] == pytest.approx(-122.87357160719932, abs=1e-10)
    assert out["e_two_electron"] == pytest.approx(37.77685108022314, abs=1e-10)
    assert out["e_total"] == pytest.approx(-76.008426802834, abs=1e-9)
    # HF: e_coulomb_plus_exchange == e_two_electron (e_xc is 0).
    assert out["e_coulomb_plus_exchange"] == pytest.approx(
        out["e_two_electron"], abs=1e-12)
    # Self-check residual is at ORCA's printing precision.
    assert abs(out["e_total_residual"]) < 1e-8
    # Closed shell: a single MO list, occupied block first.
    assert "mo_energies" in out
    assert "mo_energies_alpha" not in out
    assert out["mo_energies"][0] == pytest.approx(-20.557409, abs=1e-6)
    assert len(out["mo_energies"]) >= 5  # at least the occupied block


def test_parse_dft_sample() -> None:
    """The PBE/6-31G* H2O sample parses with a real E(XC) term.

    Pure GGA: no hybrid final integration, so the SCF-grid total
    (e_nuc + e_1e + e_2e) equals ORCA's FINAL SINGLE POINT ENERGY and
    e_final_integration_delta is 0.
    """
    out = parse_orca_output(_DFT_SAMPLE)

    assert out["code_version"] == "6.1.1"
    assert out["e_nuc"] == pytest.approx(9.0882937241424, abs=1e-10)
    assert out["e_1e"] == pytest.approx(-122.91280255103666, abs=1e-10)
    assert out["e_two_electron"] == pytest.approx(37.50407944224192, abs=1e-10)
    # DFT: E(XC) is parsed and non-zero.
    assert out["e_xc"] == pytest.approx(-9.259978237268, abs=1e-9)
    # e_coulomb_plus_exchange backs E(XC) out of the combined 2e term.
    assert out["e_coulomb_plus_exchange"] == pytest.approx(
        37.50407944224192 - (-9.259978237268), abs=1e-9)
    # Pure GGA: SCF-grid total == final total, no final integration.
    assert out["e_total"] == pytest.approx(-76.320429384652, abs=1e-9)
    assert out["e_final_integration_delta"] == 0.0
    assert out["e_total_final"] == pytest.approx(out["e_total"], abs=1e-9)
    assert abs(out["e_total_residual"]) < 1e-8


def test_parse_hybrid_final_integration() -> None:
    """A hybrid functional's finer-grid E_x recompute is accounted for.

    For a hybrid, ORCA recomputes E_x on a finer grid at the end:
    ``Nuc + 1e + 2e`` is the SCF-grid total, while ``Total Energy`` /
    ``FINAL SINGLE POINT ENERGY`` carry the post-integration value. The
    parser must return the SCF-grid total as ``e_total`` (what vibe-qc
    is comparable to) and still pass its self-check.

    Synthesised from the pure-GGA sample by injecting the
    final-integration block — keeps the test hermetic (no extra
    committed ORCA fixture) while exercising the exact hybrid code path.
    """
    import re

    text = _DFT_SAMPLE
    delta = 0.000031544  # the "Exchange energy change after final integration"
    scf_grid_total = -76.32042938465234  # e_nuc + e_1e + e_2e of the sample
    post_total = scf_grid_total + delta

    # Post-integration total on the FSP line + the block 'Total Energy'
    # (regex so the substitution is robust to ORCA's column spacing).
    text, n_fsp = re.subn(
        r"(FINAL SINGLE POINT ENERGY\s+)[-+]?\d+\.\d+",
        rf"\g<1>{post_total:.12f}", text)
    text, n_tot = re.subn(
        r"(Total Energy\s*:\s*)[-+]?\d+\.\d+(\s*Eh)",
        rf"\g<1>{post_total:.14f}\g<2>", text)
    assert n_fsp == 1 and n_tot == 1, (
        f"test fixture: expected one FSP + one Total Energy line "
        f"(got {n_fsp}, {n_tot})")
    # Inject the hybrid final-integration block ahead of the orbital table.
    text = text.replace(
        "ORBITAL ENERGIES",
        f"Exchange energy change after final integration :      "
        f"{delta:.9f} Eh\nTotal energy after final integration           :  "
        f"{post_total:.9f} Eh\n\nORBITAL ENERGIES",
        1,
    )

    out = parse_orca_output(text)
    # e_total is the SCF-grid total — unchanged from the components.
    assert out["e_total"] == pytest.approx(scf_grid_total, abs=1e-9)
    # the finer-grid correction is parsed ...
    assert out["e_final_integration_delta"] == pytest.approx(delta, abs=1e-12)
    # ... and e_total_final carries the post-integration value.
    assert out["e_total_final"] == pytest.approx(post_total, abs=1e-9)
    # self-check: (SCF-grid total + delta) reconstructs the final total.
    assert abs(out["e_total_residual"]) < 1e-8


def test_parse_accepts_text_and_path(tmp_path: Path) -> None:
    """parse_orca_output takes either a path or the raw output text."""
    sample_path = tmp_path / "orca.out"
    sample_path.write_text(_HF_SAMPLE, encoding="utf-8")
    from_path = parse_orca_output(sample_path)
    from_text = parse_orca_output(_HF_SAMPLE)
    assert from_path["e_total"] == from_text["e_total"]
    assert from_path["mo_energies"] == from_text["mo_energies"]


def test_self_check_fires_on_corrupted_total() -> None:
    """A tampered FINAL SINGLE POINT ENERGY must fail the self-check loudly.

    This is the discipline the handover demands: a wrong parse / sign /
    label drift breaks ORCA's own ``E_total = E_nuc + E_1e + E_2e``
    identity — it must raise, not feed a bad reference into the matrix.
    """
    text = _HF_SAMPLE
    # Shift the canonical total by 1 Ha — the components no longer
    # reconstruct it.
    corrupted = text.replace(
        "FINAL SINGLE POINT ENERGY       -76.008426802834",
        "FINAL SINGLE POINT ENERGY       -75.008426802834",
    )
    assert corrupted != text, "test fixture: replacement string not found"
    with pytest.raises(OrcaParseError, match="self-check FAILED"):
        parse_orca_output(corrupted)


def test_incomplete_output_raises() -> None:
    """An ORCA run that never reached the SCF total raises OrcaParseError."""
    truncated = _HF_SAMPLE.split("TOTAL SCF ENERGY")[0]
    with pytest.raises(OrcaParseError):
        parse_orca_output(truncated)


def test_missing_final_energy_raises() -> None:
    """TOTAL SCF ENERGY present but no FINAL SINGLE POINT ENERGY -> raise."""
    text = _HF_SAMPLE
    # Drop the FSP line (e.g. ORCA crashed in post-SCF) — the block is
    # still there, so this exercises a different code path than the
    # truncated-output test.
    without_fsp = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("FINAL SINGLE POINT ENERGY")
    )
    with pytest.raises(OrcaParseError, match="FINAL SINGLE POINT ENERGY"):
        parse_orca_output(without_fsp)


# --- MP2 / RI-MP2 parser ------------------------------------------------
#
# No committed ORCA MP2 sample output yet (the parity matrix's MP2
# cells + their cached ORCA references land in milestone M5). Until
# then the MP2 parser is pinned against a format-faithful synthetic
# ORCA-6.x fixture. When the M5 cache JSONs land, add a real-sample
# test alongside these — same pattern as the HF / DFT samples above.

_RIMP2_SAMPLE = """
Program Version 6.1.1  -  RELEASE  -

|  1> ! RI-MP2 cc-pVTZ/C cc-pVTZ NoFrozenCore VeryTightSCF Bohrs

----------------------------
TOTAL SCF ENERGY
----------------------------

Total Energy       :          -76.05715829 Eh           -2069.62000 eV

----------------------------------------------------------
                    ORCA  MP2 CALCULATION
----------------------------------------------------------

 Opposite-Spin pair correlation energy :     -0.18968541
 Same-Spin pair correlation energy     :     -0.05179221

 RI-MP2 CORRELATION ENERGY   :     -0.24147762 Eh
-----------------------------------------------------------
 MP2 TOTAL ENERGY:      -76.298635910 Eh
-----------------------------------------------------------

FINAL SINGLE POINT ENERGY      -76.298635910
"""


def test_parse_mp2_synthetic_sample() -> None:
    """A format-faithful synthetic RI-MP2 output parses correctly."""
    out = parse_orca_mp2_output(_RIMP2_SAMPLE)
    assert out["code"] == "orca"
    assert out["code_version"] == "6.1.1"
    assert out["e_hf"] == pytest.approx(-76.05715829, abs=1e-10)
    assert out["e_corr"] == pytest.approx(-0.24147762, abs=1e-10)
    assert out["e_total"] == pytest.approx(-76.298635910, abs=1e-10)
    assert out["e_os"] == pytest.approx(-0.18968541, abs=1e-10)
    assert out["e_ss"] == pytest.approx(-0.05179221, abs=1e-10)
    # The parser's mandatory self-check: e_hf + e_corr == e_total.
    assert abs(out["e_total_residual"]) < 1e-8
    # And the channel split sums to the correlation energy.
    assert out["e_os"] + out["e_ss"] == pytest.approx(
        out["e_corr"], abs=1e-8)


def test_parse_mp2_missing_correlation_raises() -> None:
    """An MP2 deck that never reached the post-SCF step raises."""
    scf_only = (
        "Program Version 6.1.1\n"
        "TOTAL SCF ENERGY\n"
        "Total Energy       :          -76.05715829 Eh\n"
    )
    with pytest.raises(OrcaParseError, match="MP2 CORRELATION ENERGY"):
        parse_orca_mp2_output(scf_only)


def test_parse_mp2_self_check_fires() -> None:
    """A corrupted MP2 total trips the e_hf + e_corr reconstruction check."""
    corrupted = _RIMP2_SAMPLE.replace(
        "MP2 TOTAL ENERGY:      -76.298635910 Eh",
        "MP2 TOTAL ENERGY:      -76.300000000 Eh",
    )
    assert corrupted != _RIMP2_SAMPLE, "fixture: replacement string not found"
    with pytest.raises(OrcaParseError, match="self-check FAILED"):
        parse_orca_mp2_output(corrupted)
