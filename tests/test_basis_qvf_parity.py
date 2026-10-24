"""The QVF-Basis records are the library; the text libint reads is their rendering.

``basis_library/qvf/<name>.qvf.json`` is the source of truth for every
bundled basis set (#295, #406). ``basis_library/basis/<name>.g94`` and
``<name>.ecp`` are written from the records by
``vibeqc.basis_toolkit.library.render_library`` and by nothing else, and
this module pins the two sides together:

* every rendering is byte for byte what the record renders to today, so a
  record cannot change without its text following and the text cannot be
  edited by hand;
* re-importing the rendered text gives back the record's numbers exactly
  (every exponent, coefficient, ECP term and power, as doubles), the
  round-trip pin #406 asked for before the records became authoritative;
* every record carries the 1.1.0 provenance (#407): a kind, at least one
  upstream, and a content hash that matches its content;
* every record validates against the shipped schema.

Until 2026-10 the generation ran the other way (text -> record) and had
stopped being run: fifteen records carried SP shells in a layout a since-
fixed importer produced, and 254 were named ``<g94-import>``. These checks
are text parsing and hashing, so the whole library runs in a few seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibeqc.basis_toolkit import from_g94, library
from vibeqc.basis_toolkit.model import SCHEMA_VERSION, BasisSetData

LIBRARY = library.library_root()
BASIS_DIR = library.renderings_dir()
QVF_DIR = library.records_dir()
STEMS = library.stems()


def _numbers(data: BasisSetData):
    """Everything numerical in a record, as exact Python floats and ints."""
    shells = {
        sym: [
            (tuple(s.angular_momentum), tuple((p.exponent, p.coefficient) for p in s.primitives))
            for s in el.shells
        ]
        for sym, el in data.elements.items()
    }
    ecps = {
        sym: (
            e.n_core_electrons,
            e.angular_momentum_max,
            tuple(
                (p.angular_momentum, tuple((t.power, t.exponent, t.coefficient) for t in p.terms))
                for p in e.potentials
            ),
        )
        for sym, e in (data.ecps or {}).items()
    }
    return shells, ecps


def _reimport(stem: str, text: dict[str, str]) -> BasisSetData:
    data = from_g94(text[f"{stem}.g94"], name=stem)
    if f"{stem}.ecp" in text:
        sidecar = from_g94(text[f"{stem}.ecp"], name=stem, role="ecp")
        data.ecps = dict(sidecar.ecps or {})
        for sym, element in data.elements.items():
            element.ecp_id = sym if sym in data.ecps else None
    return data


def test_the_library_is_not_empty():
    assert len(STEMS) > 200, STEMS


def test_every_rendering_has_a_record_and_no_record_lacks_one():
    """``check_library`` names a rendering without a record (a second
    original), a record without a rendering, and a rendering whose bytes
    differ; all three are empty on a consistent tree."""
    problems = library.check_library()
    assert not problems, (
        f"{len(problems)} rendering(s) out of step with the records; "
        "re-render with scripts/render_basis_library.py --target "
        f"python/vibeqc/basis_library/basis\n" + "\n".join(f"{k}: {v}" for k, v in problems.items())
    )


@pytest.mark.parametrize("stem", STEMS)
def test_rendering_is_byte_identical_to_the_record(stem):
    data = library.load_record(stem)
    rendered = library.render_record(data)
    for name, text in rendered.items():
        assert (BASIS_DIR / name).read_text(encoding="utf-8") == text, (
            f"{name} differs from the rendering of {stem}.qvf.json"
        )
    assert (BASIS_DIR / f"{stem}.ecp").is_file() == bool(data.ecps), stem


@pytest.mark.parametrize("stem", STEMS)
def test_rendered_text_reimports_to_the_record_at_full_precision(stem):
    """``repr`` writes the shortest decimal that parses back to the same
    double, so nothing is rounded on the way out (#291)."""
    data = library.load_record(stem)
    again = _reimport(stem, library.render_record(data))
    assert _numbers(again) == _numbers(data)
    assert set(again.elements) == set(data.elements)
    for sym, element in data.elements.items():
        assert (again.elements[sym].ecp_id is not None) == (element.ecp_id is not None), sym


@pytest.mark.parametrize("stem", STEMS)
def test_record_identity_and_provenance(stem):
    data = library.load_record(stem)
    assert data.name == stem
    assert data.schema_version == SCHEMA_VERSION
    prov = data.provenance
    assert prov is not None, "record without provenance"
    assert prov.kind is not None, "provenance.kind unset (#407)"
    assert prov.upstream, "no upstream recorded"
    assert prov.retrieval_date, "no retrieval date"
    assert prov.content_sha256 == data.content_sha256(), "content hash does not match the content"
    for route in prov.citations:
        assert route.keys
        if route.elements is not None:
            unknown = set(route.elements) - set(data.elements) - set(data.ecps or {})
            assert not unknown, f"citation route names elements the record lacks: {sorted(unknown)}"
    for sym, element in data.elements.items():
        assert (element.ecp_id is not None) == (sym in (data.ecps or {})), f"{sym}: ecp_id link"
    for sym, entry in (data.ecps or {}).items():
        assert entry.element == sym
        assert entry.potentials, f"{sym}: ECP without channels"
        assert max(p.angular_momentum for p in entry.potentials) == entry.angular_momentum_max, (
            f"{sym}: the local channel must be the highest one (#207)"
        )


def test_content_hash_ignores_metadata_and_tracks_numbers():
    data = library.load_record("sto-3g")
    before = data.content_sha256()
    data.description = "renamed"
    data.references = []
    assert data.content_sha256() == before
    data.elements["H"].shells[0].primitives[0].exponent *= 1.0000001
    assert data.content_sha256() != before


@pytest.fixture(scope="module")
def basis_schema_validator():
    """Use the shipped schema, requiring full JSON Schema validation."""
    import jsonschema

    schema_path = LIBRARY.parent / "basis_toolkit" / "schemas" / "qvf_basis_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_type = jsonschema.validators.validator_for(schema)
    validator_type.check_schema(schema)
    return validator_type(schema)


@pytest.mark.parametrize(
    "record_path", sorted(QVF_DIR.rglob("*.qvf.json")), ids=lambda path: path.name
)
def test_every_bundled_qvf_record_validates_against_the_shipped_schema(
    record_path, basis_schema_validator
):
    record = json.loads(record_path.read_text(encoding="utf-8"))
    errors = list(basis_schema_validator.iter_errors(record))
    assert not errors, "\n".join(
        f"{record_path.name}: /{'/'.join(map(str, error.path))}: {error.message}"
        for error in errors
    )


def test_a_one_point_zero_record_still_validates(basis_schema_validator):
    """1.1.0 only added optional provenance fields."""
    record = json.loads((QVF_DIR / "sto-3g.qvf.json").read_text(encoding="utf-8"))
    record["schema_version"] = "1.0.0"
    record["provenance"] = {"origin": "file"}
    assert not list(basis_schema_validator.iter_errors(record))
    loaded = BasisSetData.from_dict(record)
    assert loaded.provenance is not None and loaded.provenance.kind is None


@pytest.fixture
def qvf_record_with_ecp_term():
    """Select an actual bundled term so controls exercise the full record."""
    for path in sorted(QVF_DIR.rglob("*.qvf.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for ecp in record.get("ecps", {}).values():
            for potential in ecp["potentials"]:
                if potential["terms"]:
                    return record, potential["terms"][0]
    pytest.fail("No bundled ECP term found")


@pytest.mark.parametrize("power", range(5))
def test_ecp_gaussian_powers_zero_through_four_are_valid(
    power, qvf_record_with_ecp_term, basis_schema_validator
):
    record, term = qvf_record_with_ecp_term
    term["power"] = power
    assert not list(basis_schema_validator.iter_errors(record))


@pytest.mark.parametrize("power", ["2", 1.5, -1, 5, None, True])
def test_invalid_ecp_powers_are_rejected(
    power, qvf_record_with_ecp_term, basis_schema_validator
):
    record, term = qvf_record_with_ecp_term
    term["power"] = power
    errors = list(basis_schema_validator.iter_errors(record))
    assert errors
    assert all(error.path[-1] == "power" for error in errors)


def test_missing_ecp_power_is_rejected(
    qvf_record_with_ecp_term, basis_schema_validator
):
    record, term = qvf_record_with_ecp_term
    del term["power"]
    assert any(
        error.validator == "required" and "power" in error.message
        for error in basis_schema_validator.iter_errors(record)
    )


def test_an_unknown_provenance_kind_is_rejected(basis_schema_validator):
    record = json.loads((QVF_DIR / "sto-3g.qvf.json").read_text(encoding="utf-8"))
    record["provenance"]["kind"] = "downloaded"
    errors = list(basis_schema_validator.iter_errors(record))
    assert errors and all(error.path[-1] == "kind" for error in errors)


def test_a_forged_content_hash_is_rejected_by_the_schema(basis_schema_validator):
    record = json.loads((QVF_DIR / "sto-3g.qvf.json").read_text(encoding="utf-8"))
    record["provenance"]["content_sha256"] = "not-a-digest"
    errors = list(basis_schema_validator.iter_errors(record))
    assert errors and all(error.path[-1] == "content_sha256" for error in errors)


def test_render_library_writes_only_what_changed_and_prunes_strays(tmp_path):
    target = tmp_path / "basis"
    first = library.render_library(target)
    assert len(first.written) == len(list(BASIS_DIR.iterdir()))
    (target / "stray.g94").write_text("****\n")
    second = library.render_library(target)
    assert not second.written and second.removed == ["stray.g94"]
    assert library.check_library(target) == {}
    (target / "sto-3g.g94").write_text((target / "sto-3g.g94").read_text() + "!\n")
    assert library.check_library(target) == {"sto-3g.g94": "rendering differs from the record"}
