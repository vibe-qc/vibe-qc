"""Tests for vibe_basis.recipes.remote — submit recipe to vq.

All tests are mocked — no vq CLI, no CRYSTAL14, no compute-host-d.
We verify that the script emitter, submission, and result loading
work correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import pytest
from vibe_basis.recipes.pob_parity import CompoundResult, ParityReport
from vibe_basis.recipes.remote import (
    RemoteRecipeResult,
    emit_recipe_script,
    fetch_and_load_recipe_result,
    run_recipe_remote,
    submit_recipe_to_vq,
)
from vibe_basis.transports.base import (
    TERMINAL_STATES,
    JobHandle,
    JobResult,
    Transport,
    TransportError,
)
from vibe_basis.transports.vq import VqTransport

# ---------------------------------------------------------------------------
# Mock vq transport for testing
# ---------------------------------------------------------------------------


class MockVqSubmitTransport(Transport):
    """A transport that records the submit, returns a canned job id,
    and reports completed immediately.  Does NOT actually shell out
    to vq — just exercises the plumbing."""

    def __init__(self) -> None:
        self._last_handle: Optional[JobHandle] = None

    def submit(
        self,
        work_dir: Path | str,
        command: Sequence[str],
        *,
        cpus: Optional[int] = None,
        wall_time_s: Optional[int] = None,
        label: Optional[str] = None,
    ) -> JobHandle:
        self._last_handle = JobHandle(
            job_id="mock-recipe-001",
            work_dir=Path(work_dir),
            command=tuple(command),
            label=label,
        )
        return self._last_handle

    def poll(self, handle: JobHandle) -> str:
        return "completed"

    def fetch(self, handle: JobHandle, dest: Path | str) -> Path:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        out_dir = dest / handle.job_id
        out_dir.mkdir()
        (out_dir / "recipe_result.json").write_text(
            json.dumps(
                {
                    "recipe": "pob_parity",
                    "basis": "pob-tzvp",
                    "method": "rhf",
                    "compounds_total": 1,
                    "compounds_emitted": 1,
                    "compounds_converged": 1,
                    "compounds_compared": 1,
                    "sum_abs_delta_mha": 0.0,
                    "passes_acceptance": True,
                    "results": [
                        {
                            "compound": "MgO",
                            "formula": "MgO",
                            "ok": True,
                            "energy": -274.681754,
                            "method": "HF",
                            "last_cycle": 5,
                            "ref_energy": -274.681754,
                            "delta_mha": 0.0,
                            "failure_mode": None,
                            "notes": "",
                        }
                    ],
                }
            )
        )
        return out_dir


# ---------------------------------------------------------------------------
# Tests — script emission
# ---------------------------------------------------------------------------


class TestEmitRecipeScript:
    def test_emits_valid_python(self):
        """The generated script must be syntactically valid Python."""
        script = emit_recipe_script(
            recipe="pob_parity",
            structures=["PT2013-T4"],
            structure_source="table",
        )
        assert "import json" in script
        assert "from vibe_basis" in script
        assert "run_pob_parity" in script
        # Check it compiles.
        compile(script, "<generated>", "exec")

    def test_emits_with_compound_names(self):
        """Emit with structure_source="names"."""
        script = emit_recipe_script(
            structures=["MgO", "CaO"],
            structure_source="names",
        )
        # The script uses STRUCTURES[n] for lookup.
        assert "STRUCTURES[n]" in script
        # The structure_source is set to "names".
        assert '"names"' in script

    def test_emits_with_no_structures_defaults_to_all(self):
        """No structures given → the script defaults to all."""
        script = emit_recipe_script()
        assert "list(STRUCTURES.values())" in script

    def test_custom_basis_and_method_appear(self):
        script = emit_recipe_script(basis="pob-tzvp-rev2", method="pw1pw")
        assert "pob-tzvp-rev2" in script
        assert "pw1pw" in script


# ---------------------------------------------------------------------------
# Tests — submission
# ---------------------------------------------------------------------------


class TestSubmitRecipeToVq:
    def test_emits_script_and_submits(self, tmp_path):
        transport = MockVqSubmitTransport()
        handle = submit_recipe_to_vq(
            structures=["PT2013-T4"],
            vq_transport=transport,
            working_dir=tmp_path,
        )
        assert handle.job_id == "mock-recipe-001"
        # Verify the script was written.
        script_path = tmp_path / "run_recipe.py"
        assert script_path.exists()
        content = script_path.read_text()
        assert "run_pob_parity" in content

    def test_labels_are_propagated(self, tmp_path):
        transport = MockVqSubmitTransport()
        handle = submit_recipe_to_vq(
            vq_transport=transport,
            working_dir=tmp_path,
            label="my-custom-label",
        )
        assert handle.label == "my-custom-label"


# ---------------------------------------------------------------------------
# Tests — result loading
# ---------------------------------------------------------------------------


class TestFetchAndLoadRecipeResult:
    def test_fetches_and_loads(self, tmp_path):
        transport = MockVqSubmitTransport()
        handle = transport.submit(tmp_path, ["python3", "run_recipe.py"])

        result = fetch_and_load_recipe_result(
            handle,
            vq_transport=transport,
            dest_dir=tmp_path / "fetched",
        )
        assert isinstance(result, RemoteRecipeResult)
        assert result.recipe == "pob_parity"
        assert result.passes
        assert "MgO" in result.summary

    def test_raises_when_result_json_missing(self, tmp_path):
        """If the job never wrote recipe_result.json, we get a clear error."""
        transport = MockVqSubmitTransport()
        handle = transport.submit(tmp_path, ["python3", "run_recipe.py"])

        # Sabotage the fetch: write empty dir instead of the json.
        fetch_dir = tmp_path / handle.job_id
        fetch_dir.mkdir(parents=True)

        with pytest.raises(TransportError, match="recipe result not found"):
            RemoteRecipeResult.from_fetched(handle, fetch_dir)


# ---------------------------------------------------------------------------
# Tests — end-to-end convenience
# ---------------------------------------------------------------------------


class TestRunRecipeRemote:
    def test_submits_waits_fetches(self, tmp_path):
        """The convenience function chains submit → wait → fetch."""
        transport = MockVqSubmitTransport()

        # Override the working_dir so it lands in tmp_path.
        # run_recipe_remote creates its own working_dir, so let's
        # test the individual pieces instead.
        handle = submit_recipe_to_vq(
            structures=["PT2013-T4"],
            vq_transport=transport,
            working_dir=tmp_path,
        )
        state = transport.wait(handle)
        assert state == "completed"
        result = fetch_and_load_recipe_result(
            handle,
            vq_transport=transport,
            dest_dir=tmp_path / "fetched",
        )
        assert result.passes
        assert result.summary
