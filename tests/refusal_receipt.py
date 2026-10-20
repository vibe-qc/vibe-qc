"""The receipt a pre-compute capability refusal leaves under its stem (#351).

Since 5ee40dd, ``run_periodic_job`` records a ``NotImplementedError`` raised
before the job's output writer exists as a minimal ``.system`` receipt,
written through the output layer: ``run.refusal_stage = "preflight"``,
``outputs.result_status = "refused"``, and a plan that declares only the
manifest itself. Invalid input (``ValueError``, ``TypeError``) is not a
capability refusal and still writes nothing, so a test of that keeps
asserting that no ``.system`` exists.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def assert_preflight_refusal(
    stem: os.PathLike | str,
    error: BaseException | None = None,
) -> None:
    """Assert that a refused request left only its preflight refusal receipt.

    The refusal must still precede the dry-run manifest, so queue preflight
    cannot certify a request that live dispatch refuses: the receipt comes
    from the preflight stage, declares no output plan beyond itself, and is
    the only file under the stem. A supported dry run (no
    ``refusal_stage``) and a refusal raised after the output plan exists
    (full plan, other files delivered) both fail here.

    Pass the raised *error* when a parametrised test can raise either kind.
    Anything but a ``NotImplementedError`` must then have written nothing,
    and a ``NotImplementedError`` must be the one the receipt records.
    """
    stem = Path(stem)
    receipt = stem.parent / f"{stem.name}.system"
    if error is not None and not isinstance(error, NotImplementedError):
        # Invalid argument values fail before a scientific request exists.
        assert not receipt.exists()
        return
    manifest = tomllib.loads(receipt.read_text("utf-8"))
    assert manifest["run"].get("refusal_stage") == "preflight"
    outputs = manifest["outputs"]
    assert outputs["status"] == "crashed"
    assert outputs["result_status"] == "refused"
    if error is None:
        assert outputs["result_error"].startswith("NotImplementedError: ")
    else:
        assert outputs["result_error"] == f"{type(error).__name__}: {error}"
    assert [row["role"] for row in manifest["plan"]["files"]] == ["manifest"]
    delivered = sorted(
        path.name
        for path in stem.parent.iterdir()
        if path.name.startswith(f"{stem.name}.")
    )
    assert delivered == [receipt.name]
