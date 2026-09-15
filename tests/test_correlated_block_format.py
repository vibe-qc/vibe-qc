"""Format guard: correlated-method ``.out`` blocks are content-sized.

The MP2 / coupled-cluster / OVGF blocks in ``runner.py`` historically drew
hand-guessed ``"-" * 78`` rules around hand-formatted rows. They now render
through the owned primitives (``section_header`` for titled component blocks,
``Table`` for the CC iteration trace and the GF2 quasiparticle tables), so
every rule is sized to its content (CLAUDE.md Sec. 16). These pins keep the
78-dash family from returning; the numeric content itself is asserted by the
method suites (``test_runner_mp2.py``, ``test_runner_ccsd.py``,
``test_runner_ovgf.py``).
"""

from __future__ import annotations

import re

import pytest

import vibeqc as vq


@pytest.fixture()
def h2():
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )


def _out_text(tmp_path, h2, method, basis):
    stem = tmp_path / f"h2_{re.sub(r'[()]', '', method)}"
    vq.run_job(h2, basis=basis, method=method, output=stem)
    return (stem.with_suffix(".out")).read_text()


def _block(out_text: str, title_substring: str, n: int = 20) -> list[str]:
    lines = out_text.splitlines()
    for i, line in enumerate(lines):
        if title_substring in line:
            return lines[i : i + n]
    raise AssertionError(f"block {title_substring!r} not found in .out")


def _assert_no_hand_rule(out_text: str, width: int) -> None:
    """No line is exactly the old hand-guessed ``"  " + "-" * width`` rule.

    Substring checks would false-positive on longer content-sized rules
    (any rule >= width contains it), so the pin is line-exact.
    """
    hand_rule = "  " + "-" * width
    offenders = [
        i + 1 for i, l in enumerate(out_text.splitlines()) if l == hand_rule
    ]
    assert not offenders, (
        f"hand-guessed {width}-dash rule at line(s) {offenders}"
    )


def test_no_78_dash_rules_anywhere(tmp_path, h2):
    out = _out_text(tmp_path, h2, "mp2", "sto-3g")
    _assert_no_hand_rule(out, 78)
    _assert_no_hand_rule(out, 52)


def test_mp2_title_rule_is_title_sized(tmp_path, h2):
    out = _out_text(tmp_path, h2, "mp2", "sto-3g")
    title, rule = _block(out, "Moller-Plesset MP2 (")[:2]
    assert rule.strip() == "-" * len(title.strip())
    assert "E(MP2 correlation)" in out


def test_ccsd_trace_is_a_content_sized_table(tmp_path, h2):
    out = _out_text(tmp_path, h2, "ccsd", "def2-svp")
    block = _block(out, "Coupled-Cluster CCSD", n=24)
    title, title_rule = block[0], block[1]
    assert title_rule.strip() == "-" * len(title.strip())
    header = next(l for l in block if "Iter" in l and "DIIS" in l)
    header_rule = block[block.index(header) + 1]
    # The table rule fits the header exactly (indent 2 + content width).
    assert set(header_rule.strip()) == {"-"}
    assert len(header_rule) == len(header.rstrip()) or len(
        header_rule.strip()
    ) >= len(header.strip())
    _assert_no_hand_rule(out, 78)
    # dE column follows the CC-family signed-e convention (as CCSDT/CC3).
    row_1 = next(l for l in block if l.strip().startswith("1 "))
    assert re.search(r"[+-]\d\.\d{3}e[+-]\d{2}", row_1)
    assert "CCSD converged in" in out


def test_dispersion_and_thermo_blocks_are_title_sized(tmp_path, h2):
    """The 52-dash family: dispersion, vibrational, and thermochemistry
    blocks draw their rule from ``section_header`` (sized to the title)."""
    stem = tmp_path / "h2_d3_hessian"
    vq.run_job(
        h2,
        basis="sto-3g",
        method="rks",
        functional="pbe",
        dispersion="d3bj",
        hessian=True,
        output=stem,
    )
    out = stem.with_suffix(".out").read_text()
    _assert_no_hand_rule(out, 52)
    _assert_no_hand_rule(out, 78)
    # (no 60-pin here: the geometry HeaderlessBlock legitimately sizes its
    # content rule to 60 on this job; the converted 60-dash sites are the
    # MLIP card and solver frame, covered by their own suites)
    for title_sub in (
        "Dispersion correction (D3-BJ",
        "## Vibrational Frequencies",
        "## Thermochemistry (RRHO ideal gas)",
    ):
        title, rule = _block(out, title_sub)[:2]
        assert rule.strip() == "-" * len(title.strip()), title_sub


def test_ovgf_quasiparticle_table_is_content_sized(tmp_path, h2):
    out = _out_text(tmp_path, h2, "ovgf", "sto-3g")
    block = _block(out, "OVGF / GF2 quasiparticle energies")
    title, title_rule = block[0], block[1]
    assert title_rule.strip() == "-" * len(title.strip())
    header = next(l for l in block if "Koopmans(eV)" in l)
    header_rule = block[block.index(header) + 1]
    assert set(header_rule.strip()) == {"-"}
    assert "HOMO" in out and "pole Z" in header
    _assert_no_hand_rule(out, 78)
