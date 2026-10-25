"""The hand-over to vibe-qc: a Gaussian94 file and a provenance sidecar.

vibe-basis must not import vibe-qc, so the product of a run is written in
the exchange format every QC code reads, at full precision, with a JSON
sidecar that says what it is. vibe-qc's import tool reads both.
"""

from __future__ import annotations

import json

from vibe_basis.io.export import basis_to_g94, write_basis_for_import
from vibe_basis.topology import BasisTopology, Shell


def _topology() -> BasisTopology:
    return BasisTopology(
        atoms={
            "O": (
                Shell("S", (130.70932, 23.808861, 6.4436083), (0.15432897, 0.53532814, 0.44463454)),
                Shell("SP", (5.0331513, 1.1695961, 0.380389), (-0.09996723, 0.39951283, 0.70011547),
                      (0.15591627, 0.60768372, 0.39195739)),
            ),
            "Mg": (
                Shell("S", (0.1 + 0.2,), (1.0,), scale_factor=2.0),
                Shell("D", (0.04268788542106363,), (-0.0036099808435155673,)),
            ),
        }
    )


def test_g94_text_orders_elements_by_z_and_keeps_every_digit():
    text = basis_to_g94(_topology(), name="demo")
    o_at, mg_at = text.index("O     0"), text.index("Mg     0")
    assert o_at < mg_at
    assert "SP   3   1.00" in text and "0.15591627" in text
    # CRYSTAL's SCAL rule on the way out: exponent * scale_factor**2
    assert repr(0.1 + 0.2) not in text and repr((0.1 + 0.2) * 4.0) in text
    assert "0.04268788542106363" in text and "-0.0036099808435155673" in text
    assert text.count("****") == 3


def test_export_writes_the_pair_the_import_tool_reads(tmp_path):
    g94, sidecar = write_basis_for_import(
        _topology(), tmp_path, name="MgO-Opt", run_id="run-42",
        description="MgO cohesive objective", objective={"systems": ["MgO"], "mad_kjmol": 3.1},
    )
    assert g94 == tmp_path / "mgo-opt.g94" and sidecar == tmp_path / "mgo-opt.provenance.json"
    payload = json.loads(sidecar.read_text())
    assert payload["kind"] == "vibe-basis" and payload["run_id"] == "run-42"
    assert payload["name"] == "mgo-opt" and payload["objective"]["systems"] == ["MgO"]
    assert payload["generated_at"][:4].isdigit()
    head = g94.read_text().splitlines()[:3]
    assert head[0] == "! Basis: mgo-opt" and "run-42" in head[2]


def test_g94_applies_crystal_scal_squared_with_zero_and_one_as_written():
    """CRYSTAL scales a shell's exponents by SCAL**2; 0.0 and 1.0 leave them
    as written. vibe-qc's parser, converter and import tool use the same rule
    (``vibeqc.basis_crystal.crystal_exponent_factor``)."""
    from vibe_basis.io.export import crystal_exponent_factor

    assert crystal_exponent_factor(0.0) == 1.0 and crystal_exponent_factor(1.0) == 1.0
    assert crystal_exponent_factor(1.3) == 1.3 * 1.3
    topology = BasisTopology(atoms={"H": (
        Shell("S", (0.8,), (1.0,), scale_factor=1.3),
        Shell("P", (0.7,), (1.0,), scale_factor=0.0),
        Shell("SP", (0.5,), (1.0,), (1.0,), scale_factor=1.1),
    )})
    rows = [line.split() for line in basis_to_g94(topology, name="scal").splitlines()
            if line.startswith("  ")]
    assert [float(r[0]) for r in rows] == [0.8 * (1.3 * 1.3), 0.7, 0.5 * (1.1 * 1.1)]
