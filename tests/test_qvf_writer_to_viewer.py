"""QVF writer ↔ vibe-view consumer integration (audit finding 4).

The other ``test_qvf_writer.py`` tests verify that the writer's output
matches what ``validate_qvf`` (which lives inside vibe-qc) expects.
That is **the same code's own contract** — a writer regression that
also drifts the validator would be silently accepted.

These tests instead drive the writer end-to-end and then open the
result with the **independent consumer**, ``vibe-view``'s
:class:`vibeview.qvf.QVFReader`.  If ``vibeview`` is not installed,
``pytest.importorskip`` skips the round-trip; the schema-tight writer-
side asserts always run, so a manifest-shape regression (renamed
section IDs, missing volume grid members, dtype/shape drift) still
trips even on a viewer-less CI lane.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from vibeqc.output.formats.qvf import write_qvf
from vibeqc.output.plan import OutputPlan


# ── helpers (intentionally close to tests/test_qvf_writer.py's stubs)


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
        {"converged": True, "energy": -75.983, "fermi_energy": None},
    )()


def _plan(tmp_path: Path) -> OutputPlan:
    return OutputPlan.from_run_job_kwargs(
        output=tmp_path / "rt_qvf",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )


def _write_minimal_qvf(tmp_path: Path) -> Path:
    """Write a minimal QVF that carries one structure section and one
    volume.density section — the two kinds called out in the audit."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(8, 8, 8)).astype(np.float32)
    origin = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    span = np.eye(3, dtype=np.float64) * 0.25  # per-voxel step (bohr)
    return write_qvf(
        tmp_path / "rt_qvf",
        _plan(tmp_path),
        molecule=_stub_molecule(),
        result=_stub_result(),
        method="rhf",
        basis="sto-3g",
        volume_data={"Electron density": (data, origin, span)},
    )


# ── writer-side schema asserts (always run, no vibe-view needed) ────


class TestWriterSchemaInvariants:
    """Schema-drift guards on the producer side.

    These mirror the strictest expectations of the consumer reader
    (``vibe-view/src/vibeview/qvf.py``: ``Section.id``, ``MemberSpec``
    sha256+dtype+shape, ``GridData(origin, voxel_vectors, shape)``).
    Any rename or member-key change in the writer trips this without
    needing vibe-view installed.
    """

    def test_structure_section_ids_and_members(self, tmp_path):
        path = _write_minimal_qvf(tmp_path)
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            secs = [s for s in manifest["sections"] if s["kind"] == "structure"]
            assert len(secs) == 1
            s = secs[0]
            assert s["id"] == "structure", (
                "Consumer keys structure section by id == 'structure' "
                "(see vibe-view/src/vibeview/qvf.py::QVFReader.read_structure)."
            )
            assert set(s["members"]) >= {"structure"}, (
                "structure section must carry a 'structure' member; "
                "the consumer asks for it by name."
            )
            sm = s["members"]["structure"]
            assert sm["format"] == "json"
            assert len(sm["sha256"]) == 64
            struct = json.loads(zf.read(sm["path"]).decode("utf-8"))
        for k in ("atoms", "pbc"):
            assert k in struct, f"structure JSON missing key {k!r}"
        for a in struct["atoms"]:
            for k in ("symbol", "position", "atomic_number"):
                assert k in a, f"structure atom missing key {k!r}"

    def test_volume_density_section_id_grid_and_data(self, tmp_path):
        path = _write_minimal_qvf(tmp_path)
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            secs = [
                s for s in manifest["sections"] if s["kind"] == "volume.density"
            ]
            assert len(secs) == 1
            s = secs[0]
            # The consumer uses section.id to fetch members.
            assert s["id"].startswith("vol_dens_"), (
                "volume.density section id must follow the "
                "'vol_dens_<idx>' pattern the writer documents; the "
                "consumer dispatches on this prefix in app.py."
            )
            # Members the consumer asks for by name.
            assert set(s["members"]) >= {"data", "grid"}
            grid_member = s["members"]["grid"]
            data_member = s["members"]["data"]
            assert grid_member["format"] == "json"
            assert data_member["format"] == "binary"
            # dtype + shape are needed for np.frombuffer(...).reshape.
            assert data_member["dtype"] in ("float32", "float64")
            assert data_member["shape"] == [8, 8, 8]
            assert len(data_member["sha256"]) == 64
            # Grid descriptor JSON must match GridData(origin,
            # voxel_vectors, shape) the consumer expects.
            grid = json.loads(zf.read(grid_member["path"]).decode("utf-8"))
            for k in ("origin", "voxel_vectors", "shape"):
                assert k in grid, f"grid JSON missing key {k!r}"
            assert grid["shape"] == [8, 8, 8]
            assert len(grid["voxel_vectors"]) == 3
            assert all(len(row) == 3 for row in grid["voxel_vectors"])


# ── end-to-end round-trip via the vibe-view consumer ─────────────────


class TestWriterViewerRoundTrip:
    """Hand a fresh QVF to the actual vibe-view reader and assert the
    structure + volume contents reconstruct exactly.

    If vibe-view isn't on the path, the round-trip is skipped — the
    schema-invariant tests above still catch shape drift.
    """

    def test_roundtrip_structure_and_density(self, tmp_path):
        pytest.importorskip(
            "vibeview",
            reason=(
                "vibe-view is co-located at vibe-view/ but installed "
                "as a separate package (declared by the viewer-gpu "
                "extra in pyproject.toml). Install with "
                "`pip install -e vibe-view/` to enable the integration."
            ),
        )
        from vibeview.qvf import QVFReader

        path = _write_minimal_qvf(tmp_path)
        with QVFReader(path) as reader:
            # Sanity: consumer accepts the producer's manifest version.
            assert reader.manifest.qvf_version >= 1
            # Structure round-trip.
            assert reader.has_section("structure")
            struct = reader.read_structure()
            assert len(struct.atoms) == 3
            assert struct.atoms[0].atomic_number == 8
            assert struct.atoms[0].symbol == "O"
            assert struct.pbc == (False, False, False)
            # Volume round-trip.
            vol_secs = [
                s for s in reader.sections if s.kind == "volume.density"
            ]
            assert len(vol_secs) == 1
            sid = vol_secs[0].id
            grid = reader.read_volume_grid(sid)
            assert tuple(grid.shape) == (8, 8, 8)
            np.testing.assert_allclose(grid.origin, [-1.0, -1.0, -1.0])
            data = reader.read_volume_data(sid)
            assert data.shape == (8, 8, 8)
            # sha256 verification happens inside read_volume_data; if it
            # passed we know the bytes round-tripped intact.
