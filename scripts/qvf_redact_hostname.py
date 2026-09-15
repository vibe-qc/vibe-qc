#!/usr/bin/env python3
"""Audit or redact ``provenance.hostname`` in QVF archives.

An archive is meant to be handed on: attached to a paper's SI, posted on an
issue, committed as a docs example. Until v0.15.77 the documented hostname
opt-outs (``record_hostname=False`` and ``VIBEQC_NO_HOSTNAME=1``) redacted
only the ``.system`` manifest, so archives written before that fix still name
the machine they ran on. This tool finds them and, with ``--write``, masks
the field.

Only ``manifest.json`` is rewritten and every data member is copied
byte-for-byte, so the per-member sha256 values stay valid and the archive
still passes ``validate_qvf``. The field is set to ``"<redacted>"`` rather
than removed, matching the ``.system`` placeholder, so a consumer that reads
``provenance.hostname`` still finds it.

    python scripts/qvf_redact_hostname.py $(git ls-files '*.qvf')
    python scripts/qvf_redact_hostname.py --write path/to/run.qvf

Exits non-zero when any archive still carries a real host, so it can be used
as a pre-publication gate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PLACEHOLDER = "<redacted>"
# Values that are already masked by one convention or another.
_CLEAN = {None, "", PLACEHOLDER, "<host>", "example-host", "unknown"}


def audit(path: Path) -> str | None:
    """Return the recorded hostname if it is a real one, else None."""
    with zipfile.ZipFile(path) as zf:
        if "manifest.json" not in zf.namelist():
            return None
        prov = json.loads(zf.read("manifest.json")).get("provenance") or {}
    host = prov.get("hostname")
    return None if host in _CLEAN else host


def redact(path: Path) -> bool:
    """Mask the hostname in place. Returns True if the file changed."""
    if audit(path) is None:
        return False
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        infos = {i.filename: i for i in zf.infolist()}
        manifest = json.loads(zf.read("manifest.json"))
        payloads = {n: zf.read(n) for n in names if n != "manifest.json"}
    manifest.setdefault("provenance", {})["hostname"] = PLACEHOLDER

    # Write to a sibling temp file and replace atomically: a crash midway
    # must not leave a truncated archive where the original was.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".qvf.tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
            for name in names:
                if name == "manifest.json":
                    out.writestr(infos[name], json.dumps(manifest, indent=2))
                else:
                    out.writestr(infos[name], payloads[name])
        shutil.move(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument(
        "--write",
        action="store_true",
        help="mask the hostname in place (default: report only)",
    )
    args = ap.parse_args(argv)

    dirty: list[tuple[Path, str]] = []
    for path in args.paths:
        try:
            host = audit(path)
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
            print(f"  skip {path}: {type(exc).__name__}", file=sys.stderr)
            continue
        if host is None:
            continue
        dirty.append((path, host))
        if args.write:
            redact(path)
            print(f"  redacted {host!r} in {path}")
        else:
            print(f"  {host!r}  {path}")

    if not dirty:
        print("no archive carries a real hostname")
        return 0
    if args.write:
        print(f"\nredacted {len(dirty)} archive(s)")
        return 0
    print(f"\n{len(dirty)} archive(s) carry a real hostname; rerun with --write")
    return 1


if __name__ == "__main__":
    sys.exit(main())
