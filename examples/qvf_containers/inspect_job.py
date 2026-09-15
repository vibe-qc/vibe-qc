"""Inspect the lifecycle state of a pending or settled QVF job container."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import vibeqc as vq
from vibeqc.output.formats.qvf import validate_qvf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a QVF job container's spec, status, and run history."
    )
    parser.add_argument("container", type=Path)
    args = parser.parse_args()

    report = validate_qvf(args.container)
    if not report["valid"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    container = vq.load_job_container(args.container)
    with zipfile.ZipFile(args.container) as archive:
        manifest = json.loads(archive.read("manifest.json"))

    sections = manifest.get("sections", [])
    records = sorted(
        (
            section
            for section in sections
            if section.get("kind") == "run.record"
        ),
        key=lambda section: int(section.get("sequence", -1)),
    )

    print(f"path       = {args.container}")
    print("valid      = true")
    print(f"run_status = {container.run_status}")
    print(f"job_type   = {container.job_type}")
    print(f"run_count  = {len(records)}")
    print("job.spec:")
    print(json.dumps(container.spec, indent=2, sort_keys=True))
    print("section kinds:")
    for kind in sorted({str(section.get("kind")) for section in sections}):
        print(f"  {kind}")
    if records:
        print("run history:")
        for record in records:
            sequence = record.get("sequence")
            exit_status = record.get("exit_status")
            members = ", ".join(sorted(record.get("members", {})))
            print(
                f"  sequence={sequence} exit_status={exit_status} "
                f"members=[{members}]"
            )


if __name__ == "__main__":
    main()
