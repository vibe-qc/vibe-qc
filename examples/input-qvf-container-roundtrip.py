"""QVF job container round trip: pending archive in, settled archive out.

Runnable end to end with nothing but vibe-qc installed:

    python examples/input-qvf-container-roundtrip.py

It builds a *pending* container (structure + ``job.spec``,
``run_status="pending"``), runs it through the same code path
``vibeqc run water_job.qvf`` uses, and prints what the one file holds
before and after. The point is that the second archive **is** the first
file -- results, the full log, the citations and the ``.system`` manifest
are added in place and the lifecycle moves to ``converged``.

Companion tutorial: docs/tutorial/qvf_job_containers.md
Queue form (single-file payload):

    vq submit HOST water_job.qvf --program vibeqc-dev
    vq fetch  HOST JOBID --name water_job.qvf -o results/

Sample output (v0.15.60, H2O / RHF / STO-3G):

    pending  : run_status=pending      sections=['job.spec', 'structure']
    settled  : run_status=converged    E=-74.963028 Ha
    sections : atom_properties, bond_orders, citations, job.spec,
               run.record, scf_history, structure, wavefunction.gto
    log      : 137 lines embedded, identical to water_job.out
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import vibeqc as vq


def summarize(path: Path) -> dict:
    """Read the manifest's lifecycle + section list without unpacking."""
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    provenance = manifest.get("provenance", {})
    return {
        "run_status": provenance.get("run_status"),
        "energy": (provenance.get("scf_energy") or {}).get("value"),
        "sections": sorted({s["kind"] for s in manifest["sections"]}),
        "manifest": manifest,
    }


def embedded_log(path: Path) -> str | None:
    """Pull the run.record log back out of the archive, if present."""
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        record = next(
            (s for s in manifest["sections"] if s["kind"] == "run.record"),
            None,
        )
        if record is None or "log" not in record["members"]:
            return None
        return zf.read(record["members"]["log"]["path"]).decode("utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent
    stem = here / "water_job"

    # Geometry in BOHR (vibeqc.Atom's unit; QVF stores Angstrom and the
    # writer converts on the way out).
    mol = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.2217]),
            vq.Atom(1, [0.0, 1.4309, -0.8867]),
            vq.Atom(1, [0.0, -1.4309, -0.8867]),
        ]
    )

    # 1. The request, as a file. Nothing has run yet.
    container = vq.write_pending_qvf(
        mol, stem, method="rhf", basis="sto-3g"
    )
    before = summarize(container)
    print(f"pending  : run_status={before['run_status']!s:<12} "
          f"sections={before['sections']}")
    assert before["run_status"] == "pending"
    assert before["sections"] == ["job.spec", "structure"]

    # 2. Run it. The log streams as usual; the archive is updated in place.
    vq.run_container(container)

    # 3. The same file now carries the whole calculation.
    after = summarize(container)
    print(f"settled  : run_status={after['run_status']!s:<12} "
          f"E={after['energy']:.6f} Ha")
    print(f"sections : {', '.join(after['sections'])}")

    log = embedded_log(container)
    on_disk = stem.with_suffix(".out")
    same = log is not None and on_disk.exists() and (
        log == on_disk.read_text(encoding="utf-8")
    )
    print(f"log      : {len(log.splitlines()) if log else 0} lines embedded, "
          f"{'identical to' if same else 'DIFFERS from'} {on_disk.name}")

    # The request survives the run: job.spec is preserved byte-for-byte,
    # so the settled archive still shows what was asked for.
    assert "job.spec" in after["sections"]
    assert after["run_status"] == "converged"
    print(f"\nOne file holds it all: {container}")
    print("Open it with:  vibe-view open", container.name)


if __name__ == "__main__":
    main()
