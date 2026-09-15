"""Tests for the GPAW-vs-pob-vs-VASP comparison script (pure Python, no GPAW)."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from examples.regression.pw_limit_atomization.compare_pwref import (
    SYSTEM_TO_COMPOUND,
    _identify_outlier,
    _parse_gpaw_json,
    _parse_r2scan_dat,
    render_markdown,
)


def test_outlier_logic_gpaw_pob_agree_vasp_high():
    assert _identify_outlier(840, 844, 867) == "VASP"


def test_outlier_logic_gpaw_vasp_agree_pob_high():
    assert _identify_outlier(1010, 1044, 1012) == "pob"


def test_outlier_logic_all_agree():
    assert _identify_outlier(840, 842, 841) == "all agree"


def test_outlier_logic_no_consensus():
    # All three disagree substantially → no consensus
    assert _identify_outlier(800, 830, 860) == "no clear consensus"


def test_outlier_logic_gpaw_outlier():
    # pob and VASP agree, gpaw is far off
    assert _identify_outlier(900, 840, 842) == "GPAW"


def test_parse_r2scan_dat():
    with NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(r"\ce{LiF}     874.2     843.7      866.6" + "\n")
        f.write(r"\ce{MgO}     1017.4    1044.3     1012.0" + "\n")
        f.write(r"\ce{C-d}      733.4     716.6      726.2" + "\n")
        path = Path(f.name)

    try:
        rows = _parse_r2scan_dat(path)
        assert len(rows) == 3
        assert rows[0] == ("LiF", 874.2, 843.7, 866.6)
        assert rows[1] == ("MgO", 1017.4, 1044.3, 1012.0)
        assert rows[2] == ("C-d", 733.4, 716.6, 726.2)
    finally:
        path.unlink()


def test_parse_gpaw_json():
    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(
            {
                "results": [
                    {"system_id": "lif_rocksalt", "pw_limit_kjmol": 840.5},
                    {"system_id": "mgo_rocksalt", "pw_limit_kjmol": 1009.5},
                ]
            },
            f,
        )
        path = Path(f.name)

    try:
        data = _parse_gpaw_json(path)
        assert data["lif_rocksalt"] == 840.5
        assert data["mgo_rocksalt"] == 1009.5
    finally:
        path.unlink()


def test_render_markdown_generates_table():
    rows = [
        {
            "compound": "LiF",
            "E_exp": 874.2,
            "E_pob": 843.7,
            "E_VASP": 866.6,
            "E_gpaw": 840.5,
        },
        {
            "compound": "MgO",
            "E_exp": 1017.4,
            "E_pob": 1044.3,
            "E_VASP": 1012.0,
            "E_gpaw": 1009.5,
        },
    ]
    md = render_markdown(rows)
    assert "LiF" in md
    assert "MgO" in md
    assert "840.5" in md
    assert "VASP" in md  # outlier for LiF
    assert "MAD" in md
    assert "-3.2" in md  # GPAW-pob for LiF: 840.5 - 843.7 = -3.2


def test_system_to_compound_has_32_entries():
    assert len(SYSTEM_TO_COMPOUND) == 32


def test_render_markdown_with_missing_gpaw():
    rows = [
        {
            "compound": "LiF",
            "E_exp": 874.2,
            "E_pob": 843.7,
            "E_VASP": 866.6,
            "E_gpaw": None,
        },
        {
            "compound": "MgO",
            "E_exp": 1017.4,
            "E_pob": 1044.3,
            "E_VASP": 1012.0,
            "E_gpaw": 1009.5,
        },
    ]
    md = render_markdown(rows)
    # LiF should not contribute to stats (no GPAW)
    assert "MgO" in md
    assert "n=1" in md or "n= 1" in md  # only one system has GPAW data
