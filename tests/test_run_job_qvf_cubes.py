"""End-to-end ``run_job`` integration: QVF + density / MO cubes + citations.

Exercises the combination of ``output_qvf=True`` + ``write_cube=[
"density", "homo", "lumo"]`` + ``citations=True`` on a tiny H2O / RHF
/ sto-3g system. This combination was un-tested before the QVF
writer repair landed, which is how the four bugs covered there
slipped through: each individual subsystem had its own test, but
none of them exercised the integration.

After the repair, every requested artefact must land on disk:

* ``{stem}.density.cube`` (Bug 1: import path).
* ``{stem}.homo.cube`` + ``{stem}.lumo.cube`` (Bug 1 + Bug 2:
  occupation fallback for the MO label resolver).
* ``{stem}.bibtex`` + ``{stem}.references`` (Bug 3:
  ``AssembledCitations.printable``).
* ``{stem}.qvf`` carrying ``volume.density`` + two ``volume.orbital``
  sections + ``citations`` (Bug 4: int-vs-tuple index handling on
  the QVF MO data path).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest


@pytest.mark.slow
def test_run_job_with_qvf_density_homo_lumo_citations(tmp_path: Path) -> None:
    from vibeqc import Atom, Molecule, run_job

    # Standard H2O at the sto-3g validation geometry (matches
    # ``examples/h2o.xyz``, in bohr). Closed-shell, 10 electrons.
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.499211, -1.159294]),
            Atom(1, [0.0, -1.499211, -1.159294]),
        ],
        charge=0,
        multiplicity=1,
    )
    stem = tmp_path / "h2o"

    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        write_cube=["density", "homo", "lumo"],
        output_qvf=True,
        output=stem,
        citations=True,
        write_molden_file=True,
        write_population_file=True,
    )

    # --- File-on-disk checks -------------------------------------- #
    expected_targets = [
        Path(str(stem) + ".density.cube"),
        Path(str(stem) + ".homo.cube"),
        Path(str(stem) + ".lumo.cube"),
        stem.with_suffix(".bibtex"),
        stem.with_suffix(".references"),
        stem.with_suffix(".qvf"),
    ]
    for target in expected_targets:
        assert target.is_file(), f"missing artefact: {target}"
        assert target.stat().st_size > 0, f"empty artefact: {target}"

    # --- QVF manifest checks -------------------------------------- #
    qvf_path = stem.with_suffix(".qvf")
    with zipfile.ZipFile(qvf_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    kinds = [s["kind"] for s in manifest["sections"]]

    assert "structure" in kinds
    assert "volume.density" in kinds
    assert "citations" in kinds
    # write_molden_file=True must embed the basis + MO coefficients as a
    # wavefunction.gto section so vibe-view can resample any orbital on
    # demand. This used to be wired only on the periodic path; the
    # molecular runner now mirrors it.
    assert "wavefunction.gto" in kinds, (
        f"write_molden_file=True did not embed wavefunction.gto; "
        f"kinds={kinds!r}"
    )
    # Both HOMO and LUMO must produce their own ``volume.orbital``
    # section; the QVF writer emits one section per MO.
    n_orbital = sum(1 for k in kinds if k == "volume.orbital")
    assert n_orbital >= 2, (
        f"expected >= 2 volume.orbital sections (HOMO + LUMO), "
        f"got {n_orbital} in kinds={kinds!r}"
    )


@pytest.mark.slow
def test_divergent_scf_settles_the_failed_lifecycle(tmp_path: Path) -> None:
    """A genuinely non-converged SCF must settle as ``failed``.

    The ``failed`` half of the QVF lifecycle (spec § 3.2) was previously
    pinned only at the helper and writer level, because ``run_job``
    exposes no top-level iteration cap. It does expose one per method:
    ``rhf_options.max_iter``. Capping it at a single iteration produces a
    real divergent SCF through the real runner.

    What the runner does with it is deliberate and worth pinning:

    * it **raises** rather than returning, refusing to treat a
      non-converged energy as a successful calculation, so no
      ``{stem}.qvf`` result archive is written at all;
    * it still settles the lifecycle -- the checkpointer finalizes a
      terminal archive stamped ``run_status = "failed"`` -- so a consumer
      that was watching the job sees the run end, rather than a
      ``"running"`` snapshot frozen forever.
    """
    import json as _json
    import zipfile as _zipfile

    from vibeqc import Atom, Molecule, RHFOptions, run_job

    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.499211, -1.159294]),
            Atom(1, [0.0, -1.499211, -1.159294]),
        ],
        charge=0,
        multiplicity=1,
    )
    opts = RHFOptions()
    opts.max_iter = 1  # cannot converge

    stem = tmp_path / "diverged"
    checkpoint = tmp_path / "diverged_checkpoint.qvf"

    with pytest.raises(RuntimeError, match="did not converge"):
        run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            output=stem,
            output_qvf=True,
            rhf_options=opts,
            checkpoint_qvf=checkpoint,
        )

    # No result archive: the runner refuses to publish a failed energy.
    assert not stem.with_suffix(".qvf").exists(), (
        "a non-converged run must not publish a result archive"
    )

    # ... but the lifecycle is settled, not left dangling at "running".
    assert checkpoint.exists(), "no terminal checkpoint archive was written"
    with _zipfile.ZipFile(checkpoint) as zf:
        manifest = _json.loads(zf.read("manifest.json"))
    status = manifest.get("provenance", {}).get("run_status")
    assert status == "failed", (
        f"divergent SCF settled as {status!r}, expected 'failed'"
    )
