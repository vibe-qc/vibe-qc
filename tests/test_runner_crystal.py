"""Offline tests for ``examples.regression.core.runner_crystal``.

Scope: only the *pure* pieces — the .d12 deck builder, the CRYSTAL14
.out parser, the method-to-keyword mapping, the lattice-to-(a, b, c,
α, β, γ) reduction, the job-name sanitization. We do NOT invoke vq,
spawn CRYSTAL, or talk to a real host here — those are integration
concerns the laptop's regression suite skips (status='unavailable')
and Mike validates manually on compute-reference.

Fixtures are inlined synthetic CRYSTAL14 output fragments. The
anchor-line shapes match real CRYSTAL14 prints (per the .d12 +
.out conventions captured in the v0.5.x run-crystal.sh wrapper and
the reference decks under the qc-input-library checkout). When a
real .out from a compute-reference run becomes available, we can replace the
inlines with a snapshot under ``tests/data/crystal/`` without
changing the test logic.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from examples.regression.core import runner_crystal
from examples.regression.core.runner_crystal import (
    _UNSUPPORTED,
    _CRYSTAL_BASIS_LIBRARY,
    add_crystal_enecycle_keyword,
    _crystal_functional,
    _job_name,
    _lattice_to_a_b_c_alpha_beta_gamma,
    build_d12,
    parse_crystal_enecycle,
    parse_crystal_out,
)
from examples.regression.core.spec import AtomFrac, MethodSpec, PeriodicSpec


_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# build_d12 — happy paths
# ----------------------------------------------------------------------


def _mgo_spec() -> PeriodicSpec:
    """8-atom conventional MgO cell — same shape as the regression
    suite's systems/periodic/mgo_rocksalt.py."""
    a = 4.211
    mg_frac = (
        (0.0, 0.0, 0.0), (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5), (0.5, 0.5, 0.0),
    )
    o_frac = (
        (0.5, 0.5, 0.5), (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
    )
    return PeriodicSpec(
        id="mgo_rocksalt", family="rocksalt",
        lattice_ang=((a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)),
        space_group="Fm-3m",
        atoms=tuple(
            [AtomFrac("Mg", 12, p) for p in mg_frac]
            + [AtomFrac("O", 8, p) for p in o_frac]
        ),
    )


def test_build_d12_pbe_pob_tzvp() -> None:
    spec = _mgo_spec()
    method = MethodSpec(id="rks-pbe", scf="rks", xc="pbe")
    deck = build_d12(
        spec=spec, basis_kw="POB-TZVP-REV2", func_kw="PBE",
        kmesh=(8, 8, 8), conv_tol_energy=1e-7, max_iter=60,
    )
    # Header
    assert deck.startswith(
        "mgo_rocksalt :: POB-TZVP-REV2 :: PBE :: P1 (vibeqc regression)\n"
    )
    # P1 geometry block
    assert "CRYSTAL\n0 0 0\n1\n" in deck
    # Lattice line: cubic a a a 90 90 90
    assert "4.2110000 4.2110000 4.2110000 90.00000 90.00000 90.00000" in deck
    # 8-atom block: 4 Mg + 4 O. Spot-check first and last.
    assert " 12   0.0000000   0.0000000   0.0000000" in deck
    assert "  8   0.0000000   0.0000000   0.5000000" in deck
    # End-of-geometry
    assert "\nEND\nBASISSET\nPOB-TZVP-REV2\n" in deck
    # DFT block present for non-HF
    assert "\nDFT\nPBE\nXLGRID\nEND\n" in deck
    # SCF block
    assert "\nSHRINK\n8 8\n" in deck
    assert "\nTOLINTEG\n7 7 7 7 14\n" in deck
    assert "\nTOLDEE\n7\n" in deck
    assert "\nMAXCYCLE\n60\n" in deck
    # Final END
    assert deck.rstrip().endswith("END")


def test_build_d12_hf_omits_dft_block() -> None:
    spec = _mgo_spec()
    deck = build_d12(
        spec=spec, basis_kw="STO-3G", func_kw="HF",
        kmesh=(4, 4, 4), conv_tol_energy=1e-6, max_iter=80,
    )
    assert "BASISSET\nSTO-3G\nSHRINK\n4 4\n" in deck
    # No DFT block when func_kw == "HF"
    assert "DFT" not in deck
    assert "XLGRID" not in deck
    # TOLDEE derived from 1e-6 → 6
    assert "\nTOLDEE\n6\n" in deck
    assert "\nMAXCYCLE\n80\n" in deck


def test_build_d12_gamma_point_smoke() -> None:
    spec = _mgo_spec()
    deck = build_d12(
        spec=spec, basis_kw="STO-3G", func_kw="SVWN",
        kmesh=(1, 1, 1), conv_tol_energy=1e-7, max_iter=40,
    )
    # SHRINK 1 1 = Γ-point only
    assert "\nSHRINK\n1 1\n" in deck
    # SVWN gets the DFT block (LDA convention)
    assert "\nDFT\nSVWN\nXLGRID\nEND\n" in deck


def test_build_d12_anisotropic_kmesh_uses_max() -> None:
    spec = _mgo_spec()
    deck = build_d12(
        spec=spec, basis_kw="POB-TZVP-REV2", func_kw="PBE",
        kmesh=(2, 4, 8), conv_tol_energy=1e-7, max_iter=60,
    )
    # Anisotropic mesh: we pick max(k) = 8 for both SHRINK args.
    assert "\nSHRINK\n8 8\n" in deck


def test_build_d12_tight_tol_clamps_to_toldee_12() -> None:
    spec = _mgo_spec()
    deck = build_d12(
        spec=spec, basis_kw="STO-3G", func_kw="HF",
        kmesh=(1, 1, 1), conv_tol_energy=1e-20, max_iter=60,
    )
    # TOLDEE clamps at 12.
    assert "\nTOLDEE\n12\n" in deck


def test_build_d12_loose_tol_clamps_to_toldee_3() -> None:
    spec = _mgo_spec()
    deck = build_d12(
        spec=spec, basis_kw="STO-3G", func_kw="HF",
        kmesh=(1, 1, 1), conv_tol_energy=1e-1, max_iter=10,
    )
    # TOLDEE clamps at 3.
    assert "\nTOLDEE\n3\n" in deck


# ----------------------------------------------------------------------
# _lattice_to_a_b_c_alpha_beta_gamma
# ----------------------------------------------------------------------


def test_lattice_reduce_cubic() -> None:
    lat = ((4.211, 0.0, 0.0), (0.0, 4.211, 0.0), (0.0, 0.0, 4.211))
    a, b, c, al, be, ga = _lattice_to_a_b_c_alpha_beta_gamma(lat)
    assert a == pytest.approx(4.211)
    assert b == pytest.approx(4.211)
    assert c == pytest.approx(4.211)
    assert al == pytest.approx(90.0)
    assert be == pytest.approx(90.0)
    assert ga == pytest.approx(90.0)


def test_lattice_reduce_hexagonal_like() -> None:
    # Hexagonal: a = b, c distinct, γ = 120°.
    import math
    a = 2.46
    c = 6.71
    lat = (
        (a, 0.0, 0.0),
        (-a / 2, a * math.sqrt(3) / 2, 0.0),
        (0.0, 0.0, c),
    )
    a_out, b_out, c_out, al, be, ga = _lattice_to_a_b_c_alpha_beta_gamma(lat)
    assert a_out == pytest.approx(a)
    assert b_out == pytest.approx(a)
    assert c_out == pytest.approx(c)
    assert al == pytest.approx(90.0)
    assert be == pytest.approx(90.0)
    assert ga == pytest.approx(120.0)


# ----------------------------------------------------------------------
# _crystal_functional — method mapping
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, expected",
    [
        (MethodSpec(id="rks-lda", scf="rks", xc="lda"),  "SVWN"),
        (MethodSpec(id="rks-pbe", scf="rks", xc="pbe"),  "PBE"),
        (MethodSpec(id="rks-blyp", scf="rks", xc="blyp"), "BLYP"),
        # CRYSTAL's B3LYP keyword is the VWN5 flavor — exactly vibe-qc's
        # bare "b3lyp" (ORCA definition); "b3lyp5" is the explicit
        # spelling of the same flavor.
        (MethodSpec(id="rks-b3lyp", scf="rks", xc="b3lyp"), "B3LYP"),
        (MethodSpec(id="rks-b3lyp5", scf="rks", xc="b3lyp5"), "B3LYP"),
        (MethodSpec(id="rhf", scf="rhf"), "HF"),
    ],
)
def test_crystal_functional_supported(method: MethodSpec, expected: str) -> None:
    assert _crystal_functional(method) == expected


@pytest.mark.parametrize(
    "method",
    [
        # UKS/UHF: not in v1 (open-shell SPINLOCK block hasn't been wired).
        MethodSpec(id="uhf", scf="uhf"),
        MethodSpec(id="uks-pbe", scf="uks", xc="pbe"),
        # Unknown XC.
        MethodSpec(id="rks-xx", scf="rks", xc="m06-l"),
        # Gaussian-flavor B3LYP spellings: CRYSTAL14 has no VWN-RPA
        # correlation — must NOT silently map onto CRYSTAL's
        # VWN5-flavor B3LYP keyword.
        MethodSpec(id="rks-b3lypg", scf="rks", xc="b3lypg"),
        MethodSpec(id="rks-b3lyp-g", scf="rks", xc="b3lyp/g"),
        # Post-HF: not in v1 (CRYSTAL14 LCMP2 needs distinct topology).
        MethodSpec(id="mp2", scf="rhf", post="mp2"),
    ],
)
def test_crystal_functional_unsupported(method: MethodSpec) -> None:
    assert _crystal_functional(method) is _UNSUPPORTED


# ----------------------------------------------------------------------
# _CRYSTAL_BASIS_LIBRARY — basis name mapping
# ----------------------------------------------------------------------


def test_basis_library_keys() -> None:
    # The v1 whitelist. Add a key here AND extend the library mapping
    # before adding the basis to a regression case.
    assert _CRYSTAL_BASIS_LIBRARY["sto-3g"] == "STO-3G"
    assert _CRYSTAL_BASIS_LIBRARY["pob-dzvp-rev2"] == "POB-DZVP-REV2"
    assert _CRYSTAL_BASIS_LIBRARY["pob-tzvp-rev2"] == "POB-TZVP-REV2"


# ----------------------------------------------------------------------
# _job_name — vq label sanitization
# ----------------------------------------------------------------------


def test_job_name_basic() -> None:
    spec = _mgo_spec()
    method = MethodSpec(id="rks-pbe", scf="rks", xc="pbe")
    name = _job_name(spec, "pob-tzvp-rev2", method, (8, 8, 8))
    # Charset: alnum / "-" / "_" / ".".
    import re
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name)
    assert name.startswith("vqc-mgo_rocksalt-")
    assert len(name) <= 50


def test_job_name_truncates_at_50() -> None:
    spec = PeriodicSpec(
        id="a-very-long-system-identifier-that-should-be-cut",
        family="x",
        lattice_ang=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        space_group="P1",
        atoms=(AtomFrac("H", 1, (0.0, 0.0, 0.0)),),
    )
    method = MethodSpec(id="some-very-long-method-id", scf="rks", xc="pbe")
    name = _job_name(spec, "some-very-long-basis-name", method, (1, 1, 1))
    assert len(name) <= 50


# ----------------------------------------------------------------------
# parse_crystal_out — happy path + failure modes
# ----------------------------------------------------------------------

# Synthetic .out anchors mirror the real CRYSTAL14 print formats. The
# version banner is the literal phrase CRYSTAL writes at the top; the
# per-cycle and final-energy lines use CRYSTAL's fixed FORTRAN
# columns.
_CRYSTAL_SCF_CONVERGED = textwrap.dedent("""\
     *******************************************************************************
     *                                                                             *
     *                                CRYSTAL14                                    *
     *                              public : 1.0.4                                 *
     *                                                                             *
     *******************************************************************************

     [... geometry, basis, integrals ...]

     TOTAL ENERGY(DFT)(AU)(   1) -2.7400000000000E+02 DE-1.0E-01 tester 1.2E-02
     TOTAL ENERGY(DFT)(AU)(   2) -2.7491000000000E+02 DE-9.1E-03 tester 1.0E-02
     TOTAL ENERGY(DFT)(AU)(   3) -2.7494100000000E+02 DE-3.1E-04 tester 5.0E-04
     TOTAL ENERGY(DFT)(AU)(  13) -2.7494183921501E+02 DE-6.2E-09 tester 1.0E-09

     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU)  -2.74941839E+02   CYCLES  13

     DIRECT ENERGY BAND GAP:    7.4837 eV

     [...wave function dump...]
""")


_CRYSTAL_HF_CONVERGED = textwrap.dedent("""\
     *******************************************************************************
     *                                CRYSTAL17                                    *
     *******************************************************************************

     TOTAL ENERGY(HF)(AU)(   1) -7.6000000000000E+02 DE-1.0E-01 tester 1.2E-02
     TOTAL ENERGY(HF)(AU)(  12) -7.6037215834312E+02 DE-9.5E-09 tester 1.0E-09

     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU)  -7.60372158E+02   CYCLES  12

     INDIRECT ENERGY BAND GAP:    9.1023 eV
""")


_CRYSTAL_TOO_MANY_CYCLES = textwrap.dedent("""\
     *******************************************************************************
     *                                CRYSTAL14                                    *
     *******************************************************************************

     TOTAL ENERGY(DFT)(AU)(  98) -2.7488000000000E+02 DE-3.0E-03 tester 1.0E-03
     TOTAL ENERGY(DFT)(AU)(  99) -2.7488500000000E+02 DE-5.0E-04 tester 1.0E-03
     TOTAL ENERGY(DFT)(AU)( 100) -2.7488900000000E+02 DE-4.0E-04 tester 1.0E-03

     == SCF ENDED - TOO MANY CYCLES            E(AU)  -2.74889000E+02   CYCLES 100
""")


_CRYSTAL_PREFLIGHT_ERROR = textwrap.dedent("""\
     *******************************************************************************
     *                                CRYSTAL14                                    *
     *******************************************************************************

     *** GEOM *** UNRECOGNIZED SPACE GROUP NUMBER
     ABNORMAL TERMINATION OF JOB
""")


def test_parse_dft_converged_happy_path() -> None:
    res = parse_crystal_out(_CRYSTAL_SCF_CONVERGED)
    assert res.version == "CRYSTAL14"
    assert res.energy_ha == pytest.approx(-274.94183921501)
    assert res.converged is True
    assert res.n_iter == 13
    assert res.band_gap_ev == pytest.approx(7.4837)
    assert res.error_line == ""


def test_parse_hf_indirect_gap() -> None:
    res = parse_crystal_out(_CRYSTAL_HF_CONVERGED)
    assert res.version == "CRYSTAL17"
    assert res.energy_ha == pytest.approx(-760.37215834312)
    assert res.converged is True
    assert res.n_iter == 12
    assert res.band_gap_ev == pytest.approx(9.1023)


def test_parse_scf_too_many_cycles() -> None:
    res = parse_crystal_out(_CRYSTAL_TOO_MANY_CYCLES)
    # SCF did NOT converge — but we still report the last per-cycle E
    # so the regression report can show "diverged at -274.89 Ha".
    assert res.converged is False
    assert res.energy_ha == pytest.approx(-274.88900000000)
    assert res.n_iter == 100


def test_parse_preflight_error_no_energy() -> None:
    res = parse_crystal_out(_CRYSTAL_PREFLIGHT_ERROR)
    assert res.energy_ha is None
    assert res.n_iter is None
    # The first error-bracket line is captured for the row note.
    assert "GEOM" in res.error_line or "ABNORMAL" in res.error_line


def test_parse_empty_output() -> None:
    res = parse_crystal_out("")
    assert res.version is None
    assert res.energy_ha is None
    assert res.converged is None
    assert res.n_iter is None
    assert res.band_gap_ev is None


def test_parse_truncated_output_picks_last_energy() -> None:
    # Job killed mid-SCF (TIME_EXCEEDED). No terminator banner, but
    # several per-cycle energies were dumped. Parser returns the last
    # one with converged=None (no SCF terminator seen).
    truncated = (
        " *** CRYSTAL14 ***\n"
        " TOTAL ENERGY(DFT)(AU)(   1) -2.7400000000000E+02 DE-1.0E-01\n"
        " TOTAL ENERGY(DFT)(AU)(   5) -2.7492000000000E+02 DE-2.0E-04\n"
    )
    res = parse_crystal_out(truncated)
    assert res.energy_ha == pytest.approx(-274.92)
    assert res.n_iter == 5
    assert res.converged is None                  # no terminator banner seen


def test_parse_crystal_enecycle_mgo_snapshot() -> None:
    text = (
        _ROOT
        / "examples/regression/crystal_parity/crystal_demos/"
        / "mgo_sto3g_enecycle.out"
    ).read_text()

    records = parse_crystal_enecycle(text)

    assert len(records) == 8
    cyc0 = records[0]
    assert cyc0.cycle == 0
    assert cyc0.e_total == pytest.approx(-270.67396570771)
    assert cyc0.e_kinetic == pytest.approx(268.01817528309)
    assert cyc0.e_nuclear_attraction == pytest.approx(-511.81837423575)
    assert cyc0.e_nuclear_repulsion == pytest.approx(-73.084276676762)
    assert cyc0.e_bielet_zone_ee == pytest.approx(570.69557431623)
    assert cyc0.e_two_electron == pytest.approx(46.210509921712)
    assert (
        cyc0.e_bielet_zone_ee
        + cyc0.e_ext_el_pole
        + cyc0.e_ext_el_spheropole
    ) == pytest.approx(cyc0.e_two_electron, abs=1e-10)

    last = records[-1]
    assert last.cycle == 7
    assert last.e_total == pytest.approx(-271.21814374982)


def test_parse_crystal_enecycle_absent_returns_empty() -> None:
    assert parse_crystal_enecycle(_CRYSTAL_SCF_CONVERGED) == []


def test_add_crystal_enecycle_keyword_before_final_end() -> None:
    deck = "MgO\nCRYSTAL\n0 0 0\n225\n4.21\nEND\nSHRINK\n8 8\nEND\n"

    augmented = add_crystal_enecycle_keyword(deck)

    assert augmented.endswith("SHRINK\n8 8\nSETPRINT\n1\n69 999\nEND\n")
    assert add_crystal_enecycle_keyword(augmented) == augmented


# ----------------------------------------------------------------------
# vq_default_host — config.toml parse
# ----------------------------------------------------------------------


def test_vq_default_host_parses_default_host(tmp_path, monkeypatch) -> None:
    """The vq default_host parser reads from ~/.config/vq/config.toml.
    We monkeypatch ``Path.home`` to a tmp_path so the test is isolated.
    """
    cfg_dir = tmp_path / ".config" / "vq"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        'default_host = "compute-reference"\n'
        '[hosts.compute-reference]\n'
        'ssh = "compute-reference"\n'
    )
    monkeypatch.setattr(
        runner_crystal.Path, "home", lambda: tmp_path,
    )
    assert runner_crystal._vq_default_host("/usr/bin/vq") == "compute-reference"


def test_vq_default_host_missing_config_returns_none(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        runner_crystal.Path, "home", lambda: tmp_path,
    )
    assert runner_crystal._vq_default_host("/usr/bin/vq") is None


def test_vq_default_host_no_default_in_config(
    tmp_path, monkeypatch,
) -> None:
    cfg_dir = tmp_path / ".config" / "vq"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text("[hosts.foo]\nssh = \"foo\"\n")
    monkeypatch.setattr(
        runner_crystal.Path, "home", lambda: tmp_path,
    )
    assert runner_crystal._vq_default_host("/usr/bin/vq") is None
