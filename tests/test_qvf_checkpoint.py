"""Live-checkpoint QVF emission: atomic writes, status labeling, cadence.

Covers the producer-side checkpointing that :func:`vibeqc.run_job` /
:func:`vibeqc.run_periodic_job` use to let vibe-view hot-reload a *running*
job's QVF (Ask 2 of the Avogadro-parity cross-repo asks):

* ``write_qvf(atomic=True, ...)`` writes via temp-file + ``os.replace`` so a
  concurrent reader never observes a half-written zip.
* ``run_status`` / ``checkpoint`` land under ``provenance`` and validate
  against the current QVF v1 schema (no v3 bump required); ``partial``
  marks still-growing sections.
* :class:`QvfCheckpointer` drives a monotonic ``checkpoint.seq`` across a
  start frame, per-iteration cadence frames, and a terminal frame labeled
  ``"converged"`` / ``"failed"``.
* :func:`wrap_progress_for_checkpoints` transparently proxies a
  ``ProgressLogger`` and fires cadence snapshots on each ``iteration``.
"""

from __future__ import annotations

import json
import threading
import zipfile
from pathlib import Path

import pytest

from vibeqc.output.checkpoint import (
    QvfCheckpointer,
    wrap_progress_for_checkpoints,
)
from vibeqc.output.formats.qvf import (
    _mark_partial_sections,
    validate_qvf,
    write_qvf,
)
from vibeqc.output.plan import OutputPlan


# ---------------------------------------------------------------------------
# Helpers -- duck-typed stubs (mirror tests/test_qvf_round_trip.py)
# ---------------------------------------------------------------------------


def _plan(tmp_path: Path) -> OutputPlan:
    return OutputPlan.from_run_job_kwargs(
        output=tmp_path / "job", method="rhf", basis="sto-3g", functional=None
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


def _read_manifest(qvf_path: Path) -> dict:
    with zipfile.ZipFile(qvf_path) as zf:
        return json.loads(zf.read("manifest.json"))


# ---------------------------------------------------------------------------
# write_qvf: atomic write + streaming provenance fields
# ---------------------------------------------------------------------------


class TestWriteQvfStreamingFields:
    def test_run_status_and_checkpoint_validate_under_v1(self, tmp_path):
        """running/converged/failed + checkpoint land in provenance and the
        archive still passes the canonical (v1) validation gate."""
        checkpoint = {
            "seq": 7,
            "wall_time_s": 1.25,
            "written_at": "2026-07-02T10:00:00Z",
            "scf_iteration": 4,
            "energy_eh": -75.98,
        }
        p = write_qvf(
            tmp_path / "job",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            method="rhf",
            basis="sto-3g",
            run_status="running",
            checkpoint=checkpoint,
        )
        assert validate_qvf(p)["valid"]
        prov = _read_manifest(p)["provenance"]
        assert prov["run_status"] == "running"
        assert prov["checkpoint"]["seq"] == 7
        assert prov["checkpoint"]["scf_iteration"] == 4
        assert prov["checkpoint"]["energy_eh"] == pytest.approx(-75.98)

    def test_invalid_run_status_raises(self, tmp_path):
        with pytest.raises(ValueError, match="run_status"):
            write_qvf(
                tmp_path / "job",
                _plan(tmp_path),
                molecule=_stub_molecule(),
                method="rhf",
                basis="sto-3g",
                run_status="bogus",
            )

    def test_partial_marks_named_section(self, tmp_path):
        """partial_sections flags the structure section by kind, and the
        marked archive is still schema-valid."""
        p = write_qvf(
            tmp_path / "job",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            method="rhf",
            basis="sto-3g",
            run_status="running",
            checkpoint={"seq": 1, "wall_time_s": 0.0, "written_at": "t"},
            partial_sections={"structure"},
        )
        assert validate_qvf(p)["valid"]
        sections = _read_manifest(p)["sections"]
        struct = next(s for s in sections if s["kind"] == "structure")
        assert struct.get("partial") is True

    def test_mark_partial_sections_modes(self):
        base = [
            {"id": "struct", "kind": "structure"},
            {"id": "traj", "kind": "trajectory"},
        ]

        def fresh():
            return [dict(s) for s in base]

        none = fresh()
        _mark_partial_sections(none, None)
        assert all("partial" not in s for s in none)

        all_ = fresh()
        _mark_partial_sections(all_, True)
        assert all(s["partial"] is True for s in all_)

        by_kind = fresh()
        _mark_partial_sections(by_kind, {"trajectory"})
        assert by_kind[0].get("partial") is None
        assert by_kind[1]["partial"] is True

        by_id = fresh()
        _mark_partial_sections(by_id, {"struct"})
        assert by_id[0]["partial"] is True
        assert by_id[1].get("partial") is None

    def test_atomic_leaves_no_temp_files(self, tmp_path):
        write_qvf(
            tmp_path / "job",
            _plan(tmp_path),
            molecule=_stub_molecule(),
            method="rhf",
            basis="sto-3g",
            atomic=True,
        )
        leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
        assert leftovers == []

    def test_atomic_no_torn_reads_under_concurrent_reader(self, tmp_path):
        """A reader hammering the target while the writer does repeated
        atomic rewrites never sees a truncated / invalid zip."""
        plan = _plan(tmp_path)
        target = (tmp_path / "live").with_suffix(".qvf")

        def _write(seq: int):
            write_qvf(
                tmp_path / "live",
                plan,
                molecule=_stub_molecule(),
                method="rhf",
                basis="sto-3g",
                atomic=True,
                run_status="running",
                checkpoint={"seq": seq, "wall_time_s": 0.01 * seq, "written_at": "t"},
            )

        _write(0)  # seed
        stop = threading.Event()
        errors: list = []
        seqs: set = set()

        def reader():
            while not stop.is_set():
                try:
                    seqs.add(_read_manifest(target)["provenance"]["checkpoint"]["seq"])
                except Exception as exc:  # noqa: BLE001
                    errors.append(repr(exc))

        r = threading.Thread(target=reader)
        r.start()
        try:
            for seq in range(1, 40):
                _write(seq)
        finally:
            stop.set()
            r.join()

        assert errors == [], f"torn reads observed: {errors[:3]}"
        assert len(seqs) > 1  # the reader actually observed progress


# ---------------------------------------------------------------------------
# QvfCheckpointer
# ---------------------------------------------------------------------------


class TestQvfCheckpointer:
    def test_disabled_is_a_noop(self, tmp_path):
        ck = QvfCheckpointer(None, every=1, plan=_plan(tmp_path))
        assert not ck.enabled
        ck.snapshot(molecule=_stub_molecule(), method="rhf", basis="sto-3g")
        ck.finalize("converged", molecule=_stub_molecule(), method="rhf")
        assert ck.seq == 0

    def test_due_cadence(self, tmp_path):
        ck = QvfCheckpointer(tmp_path / "c", every=3, plan=_plan(tmp_path))
        assert not ck.due(0)
        assert not ck.due(1)
        assert not ck.due(2)
        assert ck.due(3)
        assert ck.due(6)
        assert not ck.due(4)

    def test_every_zero_keeps_start_and_end_only(self, tmp_path):
        ck = QvfCheckpointer(tmp_path / "c", every=0, plan=_plan(tmp_path))
        assert ck.enabled
        assert not ck.due(5)  # no cadence
        ck.snapshot(molecule=_stub_molecule(), method="rhf", basis="sto-3g")
        assert not ck.maybe_snapshot(5, molecule=_stub_molecule())
        ck.finalize("converged", molecule=_stub_molecule(), method="rhf", basis="sto-3g")
        assert ck.seq == 2  # start + terminal

    def test_sequence_monotonic_and_terminal_labeled(self, tmp_path):
        ck = QvfCheckpointer(tmp_path / "live", every=2, plan=_plan(tmp_path))
        mol = _stub_molecule()
        ck.snapshot(molecule=mol, method="rhf", basis="sto-3g")  # seq 1
        for n in range(1, 6):  # cadence at 2, 4 -> seq 2, 3
            ck.maybe_snapshot(
                n, molecule=mol, method="rhf", basis="sto-3g", energy_eh=-1.0 - n
            )
        ck.finalize("converged", molecule=mol, method="rhf", basis="sto-3g")  # seq 4

        target = (tmp_path / "live").with_suffix(".qvf")
        assert validate_qvf(target)["valid"]
        prov = _read_manifest(target)["provenance"]
        assert prov["run_status"] == "converged"
        assert prov["checkpoint"]["seq"] == 4

    def test_finalize_rejects_bad_status(self, tmp_path):
        ck = QvfCheckpointer(tmp_path / "c", every=1, plan=_plan(tmp_path))
        with pytest.raises(ValueError, match="converged"):
            ck.finalize("running", molecule=_stub_molecule())

    def test_write_failure_never_raises_and_rolls_back_seq(self, tmp_path):
        """A checkpoint write that fails is swallowed (warns) and does not
        advance the visible seq -- the calculation must not be tanked."""
        ck = QvfCheckpointer(tmp_path / "c", every=1, plan=_plan(tmp_path))
        # Force write_qvf to fail inside _write: it type-checks ``plan`` and
        # raises on a non-OutputPlan. The checkpointer must swallow + warn.
        ck._plan = object()
        with pytest.warns(RuntimeWarning, match="checkpoint"):
            ck.snapshot(molecule=_stub_molecule(), method="rhf", basis="sto-3g")
        assert ck.seq == 0


# ---------------------------------------------------------------------------
# Progress proxy (periodic per-iteration hook)
# ---------------------------------------------------------------------------


class _FakePlog:
    def __init__(self):
        self.iters = []
        self.banners = []

    def iteration(self, n, **fields):
        self.iters.append((n, fields.get("energy")))

    def banner(self, title):
        self.banners.append(title)


class TestProgressProxy:
    def test_transparent_forwarding_and_cadence(self, tmp_path):
        ck = QvfCheckpointer(tmp_path / "live", every=2, plan=_plan(tmp_path))
        real = _FakePlog()
        fired = []

        def on_iter(n, fields):
            if ck.maybe_snapshot(
                n, molecule=_stub_molecule(), method="rhf", basis="sto-3g",
                energy_eh=fields.get("energy"),
            ):
                fired.append(n)

        proxy = wrap_progress_for_checkpoints(real, on_iter)
        for n in range(1, 7):
            proxy.iteration(n, energy=-10.0 - n)
        proxy.banner("post-scf")  # non-iteration methods pass through

        assert [n for n, _ in real.iters] == [1, 2, 3, 4, 5, 6]
        assert real.banners == ["post-scf"]
        assert fired == [2, 4, 6]  # cadence every 2

    def test_wrap_noop_when_callback_falsy(self):
        real = _FakePlog()
        assert wrap_progress_for_checkpoints(real, None) is real

    def test_proxy_survives_resolve_progress(self):
        """A checkpoint-wrapped logger must pass through ``resolve_progress``
        unchanged. Every SCF driver re-normalizes its ``progress=`` arg via
        ``resolve_progress`` at the top of the function; if that returned the
        *inner* logger (the proxy is not a ``ProgressLogger`` instance), the
        per-iteration checkpoint hook would be stripped before the SCF loop
        ran and no cadence frame would ever fire -- the latent bug that made
        `checkpoint_every=N` a no-op on every route. resolve_progress passes
        through any object exposing a callable ``iteration``.
        """
        from vibeqc.progress import ProgressLogger, resolve_progress

        real = ProgressLogger(verbose=0)
        fired = []
        proxy = wrap_progress_for_checkpoints(
            real, lambda n, fields: fired.append(n)
        )
        assert not isinstance(proxy, ProgressLogger)
        resolved = resolve_progress(proxy)
        assert resolved is proxy  # not silently unwrapped to `real`
        resolved.iteration(1, energy=-1.0)
        assert fired == [1]  # the checkpoint hook still fires post-resolve

    def test_resolve_progress_still_nulls_false_and_none(self):
        """The duck-typed pass-through must not change the ``False`` / ``None``
        contract -- those still resolve to the shared silent no-op logger."""
        from vibeqc.progress import resolve_progress

        assert resolve_progress(False).level == 0
        assert resolve_progress(None).level == 0

    def test_proxy_swallows_callback_errors(self):
        real = _FakePlog()

        def boom(n, fields):
            raise RuntimeError("checkpoint blew up")

        proxy = wrap_progress_for_checkpoints(real, boom)
        proxy.iteration(1, energy=-1.0)  # must not raise
        assert real.iters == [(1, -1.0)]
