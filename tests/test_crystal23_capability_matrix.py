"""The CRYSTAL23 capability matrix (#593) stays schema-valid and rendered.

Native-free: the JSON inventory, its five-state vocabulary, issue references
and the generated block of ``docs/crystal23_capability_matrix.md`` are checked
without importing ``vibeqc``.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "crystal23_capability_matrix.py"
JSON_PATH = ROOT / "benchmarks" / "crystal23_capability_matrix.json"
DOC_PATH = ROOT / "docs" / "crystal23_capability_matrix.md"


def _module():
    spec = importlib.util.spec_from_file_location("crystal23_capability_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_json_is_valid_and_complete():
    m = _module()
    data = m.load_matrix(JSON_PATH)
    assert data["schema"] == "vibeqc.crystal23-capability-matrix.v1"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["updated"])
    assert "#593" in data["scope"]
    rows = data["rows"]
    assert len(rows) >= 50
    groups = [r["group"] for r in rows]
    # Every CRYSTAL23 Section 1.2 feature family has at least one row.
    for group in ("Hamiltonian and reference", "Dispersion and composite corrections",
                  "k sampling, occupations and SCF control",
                  "Electrostatics, exchange and truncation",
                  "Basis sets and pseudopotentials", "Periodicity and geometry",
                  "Derivatives, relaxation and lattice dynamics",
                  "Properties and wave-function analysis", "Performance and parallelism"):
        assert group in groups, group
    states = {r["state"] for r in rows}
    assert states == set(m.STATE_ORDER)
    # The parity-bearing state is a minority and every such row names its scope.
    accepted = [r for r in rows if r["state"] == "numerically-accepted"]
    assert 0 < len(accepted) < len(rows) / 4
    for r in accepted:
        assert re.search(r"\d", r["evidence"]), r["id"]
        assert r["issues"], r["id"]
    # Required #593 rows are present and not claimed as complete.
    by_id = {r["id"]: r for r in rows}
    assert by_id["rks-semilocal"]["state"] == "unverified"
    assert "#187" in by_id["rks-semilocal"]["issues"]
    assert by_id["coulomb-truncation"]["state"] == "source-tested"
    assert "#531" in by_id["coulomb-truncation"]["issues"]
    assert by_id["time-to-solution"]["state"] == "unverified"
    assert by_id["lc-hybrids-production"]["state"] == "unsupported"


def test_matrix_json_rejects_schema_drift():
    m = _module()
    data = json.loads(JSON_PATH.read_text())
    bad = json.loads(json.dumps(data))
    bad["rows"][0]["state"] = "parity"
    with pytest.raises(ValueError, match="unknown state"):
        m.validate_matrix(bad)
    bad = json.loads(json.dumps(data))
    bad["rows"][1]["issues"] = ["593"]
    with pytest.raises(ValueError, match="'#<number>'"):
        m.validate_matrix(bad)
    bad = json.loads(json.dumps(data))
    bad["rows"].append(dict(bad["rows"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        m.validate_matrix(bad)
    bad = json.loads(json.dumps(data))
    bad["states"]["accepted"] = "x"
    with pytest.raises(ValueError, match="vocabulary"):
        m.validate_matrix(bad)


def test_doc_block_matches_json_and_links_resolve():
    m = _module()
    data = m.load_matrix(JSON_PATH)
    doc = DOC_PATH.read_text(encoding="utf-8")
    assert m.splice(doc, m.render_block(data)) == doc, (
        "docs/crystal23_capability_matrix.md is stale; run "
        "scripts/crystal23_capability_matrix.py --write"
    )
    assert m.main(["--check"]) == 0
    block = doc[doc.index(m.BEGIN):doc.index(m.END)]
    for r in data["rows"]:
        assert f"| `{r['id']}` |" in block, r["id"]
    for target in re.findall(r"\]\(([^)#]+\.(?:md|json))\)", doc):
        assert (DOC_PATH.parent / target).resolve().exists(), target
