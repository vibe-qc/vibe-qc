"""End-to-end QVF round-trip + schema-drift guard tests.

These tests enforce the SSOT contract:

* **Schema identity (Rule 1).**  vibe-view's bundled `schema.json`
  has the same sha256 as the canonical
  `python/vibeqc/output/formats/qvf_manifest.schema.json`.  If anyone
  ever replaces the symlink with a divergent copy this test fails
  loudly.
* **Producer → validator → consumer round-trip (Rule 2).**
  `write_qvf` produces an archive, `validate_qvf` says it's valid
  against the canonical schema, and the vibe-view `QVFReader` opens it
  and reads back at least one structure + one volume.
* **Schema-drift guard (Rule 3).**  Every kind in
  `_IMPLEMENTED_KINDS` has a matching `oneOf` branch in the schema,
  and every `kind: {const: ...}` in the schema appears in either
  `_IMPLEMENTED_KINDS` or `_RESERVED_KINDS`.  Adding a writer without
  a schema branch (or vice versa) fails this guard.
* **Three new format features.**  `volume.difference` with operand
  cross-references, `reaction.path` self-contained waypoints, and
  `reaction.waypoints` layered over a `trajectory` all round-trip
  through the writer + validator + reader.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vibeqc.output.formats.qvf import (
    _IMPLEMENTED_KINDS,
    _RESERVED_KINDS,
    _SCHEMA_PATH,
    _load_canonical_schema,
    validate_qvf,
    write_qvf,
)
from vibeqc.output.plan import OutputPlan


# ---------------------------------------------------------------------------
# Helpers — share the stubs from test_qvf_writer
# ---------------------------------------------------------------------------


def _plan(tmp_path: Path) -> OutputPlan:
    return OutputPlan.from_run_job_kwargs(
        output=tmp_path / "rt", method="rhf", basis="sto-3g", functional=None
    )


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
                _Atom(8, (0.0, 0.0, 0.1173)),
                _Atom(1, (0.0, 1.4315, -0.9314)),
                _Atom(1, (0.0, -1.4315, -0.9314)),
            ],
            "charge": 0,
            "multiplicity": 1,
        },
    )()


def _stub_result():
    return type(
        "Result",
        (),
        {
            "converged": True,
            "energy": -75.983,
            "fermi_energy": -0.247,
            "n_iter": 3,
        },
    )()


# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------


_VIBE_VIEW_SCHEMA = (
    Path(__file__).parent.parent
    / "vibe-view"
    / "src"
    / "vibeview"
    / "schema.json"
)


class TestSchemaIdentity:
    def test_viewer_schema_matches_canonical(self):
        """vibe-view/src/vibeview/schema.json must be byte-identical to
        the canonical SSOT — either via symlink or via a deliberate
        copy. If this fails, somebody edited one and not the other.
        """
        if not _VIBE_VIEW_SCHEMA.exists():
            pytest.skip("vibe-view checkout not present")
        canonical_bytes = _SCHEMA_PATH.read_bytes()
        viewer_bytes = _VIBE_VIEW_SCHEMA.read_bytes()
        canon_sha = hashlib.sha256(canonical_bytes).hexdigest()
        view_sha = hashlib.sha256(viewer_bytes).hexdigest()
        assert canon_sha == view_sha, (
            f"vibe-view/schema.json (sha256 {view_sha[:12]}…) has "
            f"diverged from the canonical SSOT (sha256 {canon_sha[:12]}…)."
            " Re-symlink or re-copy."
        )

    def test_canonical_schema_is_valid_draft202012(self):
        import jsonschema

        schema = _load_canonical_schema()
        jsonschema.Draft202012Validator.check_schema(schema)


# ---------------------------------------------------------------------------
# Schema-drift guard — implementation list ↔ schema oneOf
# ---------------------------------------------------------------------------


def _schema_kind_consts(schema: dict) -> set[str]:
    """Walk every `kind: {const: "..."}` constraint in the schema's
    Section.oneOf and return the set of advertised section kinds."""
    found: set[str] = set()
    for branch_ref in schema["$defs"]["Section"]["oneOf"]:
        ref = branch_ref["$ref"].rsplit("/", 1)[-1]
        branch = schema["$defs"][ref]
        kind_prop = branch.get("properties", {}).get("kind", {})
        if "const" in kind_prop:
            found.add(kind_prop["const"])
        elif "pattern" in kind_prop and kind_prop["pattern"].startswith("^x_"):
            # Vendor pattern — not a single advertised kind.
            continue
    return found


class TestReactionPathLatticeIsV1:
    """`reaction.path` may carry an optional `lattice` member under
    `qvf_version: 1`.

    This is the governance ruling of 2026-07-10 (see qvf-writer/GOVERNANCE.md
    § Version history): a periodic reaction path is signalled by the *presence*
    of `lattice`, not by a manifest version bump. An optional member is the
    additive case, which must not bump `qvf_version`. These assertions pin the
    v1 schema so the capability cannot regress back behind a version bump.
    """

    @staticmethod
    def _manifest(members: dict) -> dict:
        return {
            "qvf_version": 1,
            "source": {"program": "t", "version": "1", "calculation": "neb"},
            "sections": [{"id": "rp", "kind": "reaction.path", "members": members}],
        }

    @staticmethod
    def _base() -> dict:
        h = "a" * 64
        return {
            "metadata": {"path": "m.json", "format": "json", "sha256": h},
            "coords": {"path": "c.bin", "format": "binary", "dtype": "float64",
                       "shape": [5, 3, 3], "sha256": h},
        }

    def _validator(self):
        import jsonschema

        # _SCHEMA_PATH is the canonical v1 manifest schema.
        return jsonschema.Draft202012Validator(
            json.loads(Path(_SCHEMA_PATH).read_text())
        )

    def test_v1_accepts_fixed_cell_lattice(self):
        h = "a" * 64
        lat = {"path": "l.bin", "format": "binary", "dtype": "float64",
               "shape": [3, 3], "sha256": h}
        members = {**self._base(), "lattice": lat}
        assert self._validator().is_valid(self._manifest(members))

    def test_v1_accepts_per_frame_lattice(self):
        h = "a" * 64
        lat = {"path": "l.bin", "format": "binary", "dtype": "float64",
               "shape": [5, 3, 3], "sha256": h}
        members = {**self._base(), "lattice": lat}
        assert self._validator().is_valid(self._manifest(members))

    def test_v1_molecular_path_without_lattice_still_valid(self):
        assert self._validator().is_valid(self._manifest(self._base()))

    def test_lattice_must_be_float64(self):
        h = "a" * 64
        lat = {"path": "l.bin", "format": "binary", "dtype": "float32",
               "shape": [3, 3], "sha256": h}
        members = {**self._base(), "lattice": lat}
        assert not self._validator().is_valid(self._manifest(members))


class TestSchemaDriftGuard:
    def test_every_writer_kind_has_a_schema_branch(self):
        """Every kind in `_IMPLEMENTED_KINDS` must appear as a
        `kind: {const: ...}` branch in the schema. Adding a writer
        without adding a schema branch breaks this guard.
        """
        schema = _load_canonical_schema()
        schema_kinds = _schema_kind_consts(schema)
        missing = _IMPLEMENTED_KINDS - schema_kinds
        assert not missing, (
            "kinds in _IMPLEMENTED_KINDS missing a schema branch: "
            f"{sorted(missing)}"
        )

    def test_every_schema_branch_has_a_registry_entry(self):
        """Every advertised `kind: {const: ...}` in the schema must
        appear in `_IMPLEMENTED_KINDS` or `_RESERVED_KINDS`. Adding a
        schema branch without bumping the registry breaks this guard.
        """
        schema = _load_canonical_schema()
        schema_kinds = _schema_kind_consts(schema)
        registered = _IMPLEMENTED_KINDS | _RESERVED_KINDS
        missing = schema_kinds - registered
        assert not missing, (
            "kinds in the schema's Section.oneOf missing from the "
            f"writer registry: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# End-to-end round-trip — writer → validator → vibe-view reader
# ---------------------------------------------------------------------------


def _vibeview_importable() -> bool:
    """vibe-view is a separate package; only run reader-side tests if
    it's importable from the current environment."""
    try:
        import vibeview.qvf  # noqa: F401
    except ImportError:
        return False
    return True


class TestRoundTrip:
    @pytest.mark.skipif(
        not _vibeview_importable(),
        reason="vibe-view not installed in this environment",
    )
    def test_writer_validator_viewer_round_trip(self, tmp_path):
        from vibeview.qvf import QVFReader

        density = np.random.randn(8, 8, 8).astype(np.float32)
        path = write_qvf(
            tmp_path / "rt_qvf",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            volume_data={
                "rho": (density, np.array([-2.0, -2.0, -2.0]), np.eye(3) * 0.5),
            },
        )

        # 1. write-time gate already validated; double-check we agree.
        report = validate_qvf(path)
        assert report["valid"], report["errors"]

        # 2. vibe-view's reader opens the archive without raising.
        with QVFReader(path) as reader:
            assert reader.manifest.qvf_version == 1
            assert reader.has_section("structure")
            struct = reader.read_structure()
            assert len(struct.atoms) == 3
            assert struct.atoms[0].symbol == "O"

            # 3. The density section is readable through the reader's
            #    typed accessor; values match what the writer fed in.
            dens_sec = next(
                s for s in reader.sections if s.kind == "volume.density"
            )
            data = reader.read_volume_data(dens_sec.id)
            np.testing.assert_allclose(data.ravel(), density.ravel())

    @pytest.mark.skipif(
        not _vibeview_importable(),
        reason="vibe-view not installed in this environment",
    )
    def test_scf_history_round_trips_through_reader(self, tmp_path):
        """Writer-emitted scf_history is readable by vibe-view's reader,
        and the iteration count reaches the provenance block."""
        from vibeview.qvf import QVFReader

        history = [
            {"iter": 1, "energy_eh": -75.0},
            {"iter": 2, "energy_eh": -75.9, "delta_e": -0.9},
            {"iter": 3, "energy_eh": -75.983, "delta_e": -0.083},
        ]
        path = write_qvf(
            tmp_path / "hist_rt",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            scf_history_data=history,
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with QVFReader(path) as reader:
            sec = next(s for s in reader.sections if s.kind == "scf_history")
            hist = reader.read_scf_history(sec.id)
            assert len(hist.iterations) == 3
            assert hist.iterations[-1]["energy_eh"] == pytest.approx(-75.983)
            # Iteration count surfaces in provenance for the Run Info panel.
            assert reader.manifest.provenance.get("n_scf_iterations") == 3


# ---------------------------------------------------------------------------
# New format features — volume.difference / reaction.path / reaction.waypoints
# ---------------------------------------------------------------------------


class _Mol(_stub_molecule().__class__):
    pass  # so we can construct frame copies that satisfy the same shape


def _frames(n: int):
    mol = _stub_molecule()
    return [mol] * n


class TestVolumeDifferenceRoundTrip:
    def test_two_densities_plus_difference_with_operand_refs(self, tmp_path):
        rho_a = np.random.randn(6, 6, 6).astype(np.float32)
        rho_b = np.random.randn(6, 6, 6).astype(np.float32)
        diff = rho_a - rho_b
        origin = np.zeros(3)
        span = np.eye(3) * 0.5
        path = write_qvf(
            tmp_path / "diff_qvf",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            volume_data={"rho_a": (rho_a, origin, span), "rho_b": (rho_b, origin, span)},
            diff_data={
                "rho_a_minus_rho_b": {
                    "data": diff,
                    "origin": origin,
                    "span": span,
                    "operand_a": "vol_dens_0",
                    "operand_b": "vol_dens_1",
                    "description": "ρ(a) − ρ(b)",
                },
            },
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]

        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            diff_sec = next(s for s in mf["sections"] if s["kind"] == "volume.difference")
            assert diff_sec["operand_a"] == "vol_dens_0"
            assert diff_sec["operand_b"] == "vol_dens_1"
            assert diff_sec["description"] == "ρ(a) − ρ(b)"

    def test_spectra_epr_round_trip(self, tmp_path):
        """The canonical spectra.epr section writes, validates, and carries
        the g-tensor / hyperfine / zero-field-splitting payload verbatim."""
        epr = {
            "g_tensor": {
                "principal": [2.0023, 2.0021, 2.0089],
                "isotropic": 2.0044,
            },
            "hyperfine": [
                {"atom_index": 0, "symbol": "N", "isotope": "14N", "a_iso_mhz": 45.2}
            ],
            "zero_field_splitting": {"d_mhz": 1200.0, "e_mhz": 30.0},
        }
        path = write_qvf(
            tmp_path / "epr_qvf",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            epr_data=epr,
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]

        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            epr_sec = next(s for s in mf["sections"] if s["kind"] == "spectra.epr")
            payload = json.loads(zf.read(epr_sec["members"]["spectrum"]["path"]))
            assert payload["g_tensor"]["principal"] == [2.0023, 2.0021, 2.0089]
            assert payload["zero_field_splitting"]["d_mhz"] == 1200.0

    def test_volume_difference_with_unresolved_operand_fails_validation(self, tmp_path):
        """The cross-reference checker rejects archives whose
        `operand_a` / `operand_b` don't name a section in the same
        archive — even though jsonschema alone can't catch this."""
        diff = np.random.randn(4, 4, 4).astype(np.float32)
        origin = np.zeros(3)
        span = np.eye(3) * 0.5
        with pytest.raises(ValueError, match="canonical validation"):
            write_qvf(
                tmp_path / "bad_diff_qvf",
                _plan(tmp_path),
                molecule=_stub_molecule(),
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                diff_data={
                    "orphan": {
                        "data": diff,
                        "origin": origin,
                        "span": span,
                        "operand_a": "no_such_section",
                        "operand_b": "also_missing",
                    },
                },
            )

    def test_volume_difference_with_only_one_operand_fails(self, tmp_path):
        """Schema's dependentRequired: setting only one of operand_a /
        operand_b is rejected."""
        diff = np.random.randn(4, 4, 4).astype(np.float32)
        origin = np.zeros(3)
        span = np.eye(3) * 0.5
        with pytest.raises(ValueError, match="canonical validation"):
            write_qvf(
                tmp_path / "half_diff_qvf",
                _plan(tmp_path),
                molecule=_stub_molecule(),
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                volume_data={"rho_a": (diff, origin, span)},  # creates vol_dens_0
                diff_data={
                    "half": {
                        "data": diff,
                        "origin": origin,
                        "span": span,
                        "operand_a": "vol_dens_0",
                        # operand_b missing
                    },
                },
            )


class TestReactionPathRoundTrip:
    def test_self_contained_reaction_path(self, tmp_path):
        frames = _frames(5)
        waypoints = [
            {"frame_index": 0, "label": "reactant", "kind": "reactant", "energy_eh": -75.5},
            {"frame_index": 2, "label": "TS", "kind": "transition_state", "energy_eh": -75.4},
            {"frame_index": 4, "label": "product", "kind": "product", "energy_eh": -75.6},
        ]
        path = write_qvf(
            tmp_path / "rxn_qvf",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            reaction_path={
                "frames": frames,
                "waypoints": waypoints,
                "energies": [-75.5, -75.45, -75.4, -75.5, -75.6],
                "reaction_coordinate": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            rxn = next(s for s in mf["sections"] if s["kind"] == "reaction.path")
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            assert len(meta["waypoints"]) == 3
            assert {w["kind"] for w in meta["waypoints"]} == {
                "reactant",
                "transition_state",
                "product",
            }
            assert meta["reaction_coordinate"] == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_waypoint_with_out_of_range_frame_index_raises(self, tmp_path):
        """The writer's `_validate_waypoints` rejects malformed
        waypoints before zipping — the schema can't enforce this
        (frame_index is opaque to the manifest schema)."""
        frames = _frames(3)
        with pytest.raises(ValueError, match="outside"):
            write_qvf(
                tmp_path / "bad_rxn",
                _plan(tmp_path),
                molecule=_stub_molecule(),
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                reaction_path={
                    "frames": frames,
                    "waypoints": [
                        {"frame_index": 99, "label": "TS", "kind": "transition_state"},
                    ],
                },
            )

    def test_waypoint_with_unregistered_kind_raises(self, tmp_path):
        frames = _frames(3)
        with pytest.raises(ValueError, match="not one of"):
            write_qvf(
                tmp_path / "bad_kind",
                _plan(tmp_path),
                molecule=_stub_molecule(),
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                reaction_path={
                    "frames": frames,
                    "waypoints": [
                        {"frame_index": 0, "label": "?", "kind": "ts"},
                    ],
                },
            )


class TestReactionWaypointsRoundTrip:
    def test_waypoints_layered_over_trajectory(self, tmp_path):
        frames = _frames(4)
        path = write_qvf(
            tmp_path / "wp_qvf",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            trajectory_frames=frames,
            trajectory_energies=[-75.5, -75.4, -75.3, -75.6],
            reaction_waypoints={
                "trajectory_ref": "traj0",
                "waypoints": [
                    {"frame_index": 0, "label": "start", "kind": "reactant"},
                    {"frame_index": 2, "label": "barrier", "kind": "transition_state"},
                    {"frame_index": 3, "label": "end", "kind": "product"},
                ],
            },
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            kinds = [s["kind"] for s in mf["sections"]]
            assert "trajectory" in kinds
            wp_sec = next(s for s in mf["sections"] if s["kind"] == "reaction.waypoints")
            assert wp_sec["trajectory_ref"] == "traj0"

    def test_waypoints_with_unresolved_trajectory_ref_raises(self, tmp_path):
        with pytest.raises(ValueError, match="trajectory_ref"):
            write_qvf(
                tmp_path / "no_traj",
                _plan(tmp_path),
                molecule=_stub_molecule(),
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                reaction_waypoints={
                    "trajectory_ref": "nonexistent",
                    "waypoints": [
                        {"frame_index": 0, "label": "x", "kind": "point"},
                    ],
                },
            )


class TestViewerDefaultsBookmarks:
    def test_bookmarks_round_trip(self, tmp_path):
        path = write_qvf(
            tmp_path / "bm_qvf",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            result=_stub_result(),
            method="rhf",
            basis="sto-3g",
            viewer_defaults={
                "auto_open": ["structure"],
                "bookmarks": [
                    {
                        "name": "front",
                        "camera": {
                            "position": [0.0, 0.0, 10.0],
                            "focal_point": [0.0, 0.0, 0.0],
                            "view_up": [0.0, 1.0, 0.0],
                            "view_angle": 30.0,
                        },
                    },
                    {
                        "name": "side",
                        "camera": {
                            "position": [10.0, 0.0, 0.0],
                            "focal_point": [0.0, 0.0, 0.0],
                            "view_up": [0.0, 1.0, 0.0],
                            "parallel_scale": 5.0,
                        },
                    },
                ],
            },
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            bm = mf["viewer_defaults"]["bookmarks"]
            assert [b["name"] for b in bm] == ["front", "side"]
            assert "view_angle" in bm[0]["camera"]
            assert "parallel_scale" in bm[1]["camera"]

    def test_bookmark_with_both_view_angle_and_parallel_scale_fails(self, tmp_path):
        """Schema's oneOf: a camera must specify exactly one of
        `view_angle` or `parallel_scale` (perspective vs orthographic).
        Setting both is rejected by the write-time validation gate.
        """
        with pytest.raises(ValueError, match="canonical validation"):
            write_qvf(
                tmp_path / "bad_cam",
                _plan(tmp_path),
                molecule=_stub_molecule(),
                result=_stub_result(),
                method="rhf",
                basis="sto-3g",
                viewer_defaults={
                    "bookmarks": [
                        {
                            "name": "bad",
                            "camera": {
                                "position": [0, 0, 1],
                                "focal_point": [0, 0, 0],
                                "view_up": [0, 1, 0],
                                "view_angle": 30.0,
                                "parallel_scale": 5.0,
                            },
                        },
                    ],
                },
            )
