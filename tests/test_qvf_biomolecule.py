"""Biomolecule metadata on the QVF structure section (Ask B).

chains / residues / secondary_structure / b_factors are optional peer
keys on the ``structure`` section object — the additive case, so no
qvf_version bump (qvf-writer/GOVERNANCE.md versioning policy; same
philosophy as the landed live-checkpoint fields, commit 1eb02de1). As of
2026-07-26 they are no longer merely riding an open Section object: they
are schema'd as optional properties of ``$defs/SectionStructure``.

These tests assert the fields write, validate WITHOUT a qvf_version bump,
round-trip values including numpy inputs, that a structure carrying NONE
of them still validates, and that the schema now actually constrains the
shapes it names.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from vibeqc.output.formats.qvf import (
    _load_canonical_schema,
    _normalize_biomolecule,
    validate_qvf,
    write_qvf,
)
from vibeqc.output.plan import OutputPlan


def _plan(tmp_path: Path, **overrides) -> OutputPlan:
    kwargs = dict(
        output=tmp_path / "test_qvf", method="rhf", basis="sto-3g", functional=None
    )
    kwargs.update(overrides)
    return OutputPlan.from_run_job_kwargs(**kwargs)


def _stub_molecule():
    class _Atom:
        def __init__(self, Z, xyz):
            self.Z = Z
            self.xyz = xyz

    return type(
        "Molecule",
        (),
        {
            "atoms": [
                _Atom(7, (0.0, 0.0, 0.0)),
                _Atom(6, (0.0, 0.0, 1.5)),
                _Atom(8, (0.0, 1.2, 2.2)),
            ],
            "charge": 0,
            "multiplicity": 1,
        },
    )()


def _stub_result():
    return type(
        "Result",
        (),
        {"converged": True, "energy": -1.0, "n_iter": 1, "scf_trace": ()},
    )()


def _structure_section(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        qvf_version = manifest["qvf_version"]
        section = next(s for s in manifest["sections"] if s["kind"] == "structure")
    return qvf_version, section


def _minimal_manifest(**structure_extras) -> dict:
    """A root-valid manifest whose one structure section carries ``extras``.

    Used to exercise the schema directly, without writing an archive: the
    point is what the SSOT accepts and rejects on the section object.
    """
    section = {
        "id": "structure",
        "kind": "structure",
        "members": {
            "structure": {
                "path": "structure/structure.json",
                "format": "json",
                "sha256": "0" * 64,
            }
        },
    }
    section.update(structure_extras)
    return {
        "qvf_version": 1,
        "source": {"program": "test", "version": "0", "calculation": "unit"},
        "sections": [section],
    }


class TestBiomoleculeWrite:
    def test_all_four_fields_write_and_validate_without_version_bump(self, tmp_path):
        bio = {
            "chains": ["A", "B"],
            "residues": [
                {"name": "GLY", "seq": 1, "chain": "A", "atom_indices": [0, 1]},
                {"name": "ALA", "seq": 2, "chain": "A", "atom_indices": [2]},
            ],
            "secondary_structure": [
                {"type": "helix", "chain": "A", "start_seq": 1, "end_seq": 2},
            ],
            "b_factors": [12.5, 30.0, 8.25],
        }
        path = write_qvf(
            tmp_path / "bio",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            biomolecule_data=bio,
        )
        # write_qvf runs the validate_qvf gate internally; an explicit
        # re-validate documents that the additive fields pass the SSOT.
        validate_qvf(path)

        qvf_version, section = _structure_section(path)
        # Additive: no version bump.
        assert qvf_version == 1
        # Peer keys on the section object (not inside structure.json).
        assert section["chains"] == ["A", "B"]
        assert section["residues"][0] == {
            "name": "GLY",
            "seq": 1,
            "chain": "A",
            "atom_indices": [0, 1],
        }
        assert section["secondary_structure"][0]["type"] == "helix"
        assert section["b_factors"] == [12.5, 30.0, 8.25]

    def test_b_factors_alone_write_and_validate(self, tmp_path):
        path = write_qvf(
            tmp_path / "bfac_only",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            biomolecule_data={"b_factors": np.array([1.0, 2.0, 3.0])},
        )
        validate_qvf(path)
        _, section = _structure_section(path)
        assert section["b_factors"] == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in section["b_factors"])
        assert "chains" not in section
        assert "residues" not in section

    def test_fields_are_independently_optional(self, tmp_path):
        # Only chains supplied — residues/secondary_structure absent entirely.
        path = write_qvf(
            tmp_path / "chains_only",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            biomolecule_data={"chains": ["A"]},
        )
        _, section = _structure_section(path)
        assert section["chains"] == ["A"]
        assert "residues" not in section
        assert "secondary_structure" not in section

    def test_no_biomolecule_data_leaves_section_unchanged(self, tmp_path):
        path = write_qvf(
            tmp_path / "plain",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
        )
        # An older reader must be able to ignore these fields, so a file
        # carrying none of them has to stay a valid plain structure.
        validate_qvf(path)
        _, section = _structure_section(path)
        assert "chains" not in section
        assert "residues" not in section
        assert "secondary_structure" not in section
        assert "b_factors" not in section

    def test_numpy_inputs_are_coerced_and_round_trip(self, tmp_path):
        # numpy ints/arrays must survive safe_json_bytes + validate_qvf.
        bio = {
            "chains": np.array(["A", "B"]),
            "residues": [
                {
                    "name": "SER",
                    "seq": np.int64(7),
                    "chain": "B",
                    "atom_indices": np.array([0, 2], dtype=np.int64),
                },
            ],
            "secondary_structure": [
                {
                    "type": "sheet",
                    "chain": "B",
                    "start_seq": np.int64(7),
                    "end_seq": np.int32(9),
                },
            ],
        }
        path = write_qvf(
            tmp_path / "numpy_bio",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            biomolecule_data=bio,
        )
        validate_qvf(path)
        _, section = _structure_section(path)
        res = section["residues"][0]
        assert res["seq"] == 7 and isinstance(res["seq"], int)
        assert res["atom_indices"] == [0, 2]
        assert all(isinstance(i, int) for i in res["atom_indices"])
        ss = section["secondary_structure"][0]
        assert ss["start_seq"] == 7 and ss["end_seq"] == 9
        assert section["chains"] == ["A", "B"]


class TestNormalizeBiomolecule:
    def test_none_and_non_dict_yield_empty(self):
        assert _normalize_biomolecule(None) == {}
        assert _normalize_biomolecule([1, 2, 3]) == {}
        assert _normalize_biomolecule("nope") == {}

    def test_malformed_entries_dropped_not_raised(self):
        out = _normalize_biomolecule(
            {
                "chains": ["A"],
                "residues": ["not-a-dict", {"name": "X", "seq": 1, "chain": "A"}],
                "secondary_structure": [42],
            }
        )
        assert out["chains"] == ["A"]
        # the string residue is dropped; the dict one keeps defaults
        assert len(out["residues"]) == 1
        assert out["residues"][0]["atom_indices"] == []
        # the int SS entry is dropped
        assert out["secondary_structure"] == []

    def test_unknown_secondary_structure_type_is_dropped(self):
        # The schema pins the enum, so writing a DSSP "turn" would fail the
        # write-time validate gate and take the whole archive down. Drop the
        # one range instead.
        out = _normalize_biomolecule(
            {
                "secondary_structure": [
                    {"type": "turn", "chain": "A", "start_seq": 1, "end_seq": 3},
                    {"type": "sheet", "chain": "A", "start_seq": 4, "end_seq": 9},
                ]
            }
        )
        assert [s["type"] for s in out["secondary_structure"]] == ["sheet"]

    def test_non_numeric_b_factor_is_neutralized_in_place(self):
        # Dropping it would shift every later atom's b-factor by one.
        out = _normalize_biomolecule({"b_factors": [1.0, "n/a", np.float32(3.5)]})
        assert out["b_factors"] == [1.0, 0.0, 3.5]

    def test_b_factors_absent_when_not_supplied(self):
        assert "b_factors" not in _normalize_biomolecule({"chains": ["A"]})


class TestSchemaConstrainsBiomoleculeFields:
    """The fields are now named in the SSOT, not merely tolerated by it."""

    def _validate(self, manifest) -> None:
        jsonschema.Draft202012Validator(_load_canonical_schema(1)).validate(manifest)

    def test_plain_structure_without_any_field_validates(self):
        self._validate(_minimal_manifest())

    def test_well_formed_fields_validate(self):
        self._validate(
            _minimal_manifest(
                chains=["A"],
                residues=[
                    {"name": "GLY", "seq": 1, "chain": "A", "atom_indices": [0, 1]}
                ],
                secondary_structure=[
                    {"type": "coil", "chain": "A", "start_seq": 1, "end_seq": 1}
                ],
                b_factors=[10.0, 11.5],
            )
        )

    def test_residue_missing_required_key_is_rejected(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate(
                _minimal_manifest(
                    residues=[{"name": "GLY", "seq": 1, "atom_indices": [0]}]
                )
            )

    def test_unknown_secondary_structure_type_is_rejected(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate(
                _minimal_manifest(
                    secondary_structure=[
                        {"type": "turn", "chain": "A", "start_seq": 1, "end_seq": 3}
                    ]
                )
            )

    def test_non_numeric_b_factors_entry_is_rejected(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate(_minimal_manifest(b_factors=[1.0, "n/a"]))

    def test_chains_must_be_strings(self):
        with pytest.raises(jsonschema.ValidationError):
            self._validate(_minimal_manifest(chains=[1, 2]))

    def test_per_atom_b_factor_is_typed_in_structure_payload(self):
        schema = _load_canonical_schema(1)
        atom_def = schema["$defs"]["StructureAtom"]
        assert atom_def["properties"]["b_factor"]["type"] == "number"
        validator = jsonschema.Draft202012Validator(
            {**atom_def, "$defs": schema["$defs"]}
        )
        validator.validate(
            {"symbol": "C", "position": [0, 0, 0], "atomic_number": 6, "b_factor": 9.9}
        )
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(
                {
                    "symbol": "C",
                    "position": [0, 0, 0],
                    "atomic_number": 6,
                    "b_factor": "high",
                }
            )
