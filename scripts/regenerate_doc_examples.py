#!/usr/bin/env python3
"""Regenerate docs/_static/examples artifacts from canonical examples.

The script is deliberately usable both in the checkout and inside a
vq-submitted payload. It runs each selected example from its own
directory, captures stdout/stderr, copies the input script and all
declared generated artifacts into docs/_static/examples/<slug>/, and
can add vibe-view screenshots from already generated QVF archives.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO / "examples"
DEFAULT_MANIFEST = REPO / "scripts" / "doc_examples.toml"
DEFAULT_OUTPUT_ROOT = REPO / "docs" / "_static" / "examples"


@dataclass(frozen=True)
class Job:
    script: str
    slug: str
    description: str
    output_globs: tuple[str, ...]
    qvf_expected: bool = True


def load_jobs(path: Path) -> list[Job]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    jobs = []
    for item in data.get("job", []):
        jobs.append(
            Job(
                script=str(item["script"]),
                slug=str(item["slug"]),
                description=str(item["description"]),
                output_globs=tuple(str(p) for p in item.get("output_globs", ())),
                qvf_expected=bool(item.get("qvf_expected", True)),
            )
        )
    return jobs


def resolve_script(name: str) -> Path:
    hits = sorted(EXAMPLES_DIR.rglob(name))
    hits = [p for p in hits if p.is_file()]
    if not hits:
        raise FileNotFoundError(f"example script not found: {name}")
    if len(hits) > 1:
        choices = ", ".join(str(p.relative_to(REPO)) for p in hits)
        raise RuntimeError(f"ambiguous script {name}: {choices}")
    return hits[0]


def clean_declared_outputs(script_dir: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in script_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def sanitize_text_artifact(path: Path) -> None:
    """Remove local checkout, queue, and home paths from public text artifacts."""
    if path.suffix.lower() not in {
        ".txt",
        ".out",
        ".system",
        ".json",
        ".md",
        ".bibtex",
        ".references",
        ".csv",
        ".hess",
        ".engrad",
        ".inp",
    }:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return

    replacements = {str(REPO): "<vibe-qc-checkout>"}
    for before, after in replacements.items():
        text = text.replace(before, after)

    checkout_patterns = [
        (r"/Users/[^/\s]+/gitlab/vibeqc-documentation", "<vibe-qc-checkout>"),
        (r"/home/[^/\s]+/gitlab/vibeqc-dev", "<vibe-qc-checkout>"),
        (r"/home/[^/\s]+/gitlab/vibeqc-release", "<vibe-qc-release-checkout>"),
        (r"/home/[^/\s]+/gitlab/vibeqc-queue", "<vibe-qc-queue-checkout>"),
    ]
    for pattern, replacement in checkout_patterns:
        text = re.sub(pattern, replacement, text)

    text = re.sub(
        r"/var/lib/vq/users/\d+/jobs/[0-9a-f]{12}",
        "<vq-workspace>",
        text,
    )
    text = re.sub(
        r"/var/lib/vq/users/\d+/workdirs/[0-9a-f]{12}",
        "<vq-workdir>",
        text,
    )
    text = re.sub(r"/Users/[^/\s]+/", "~/", text)
    text = re.sub(r"/home/[^/\s]+/", "~/", text)
    path.write_text(text, encoding="utf-8")


def run_one(job: Job, output_root: Path) -> dict[str, object]:
    script = resolve_script(job.script)
    target = output_root / job.slug
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    clean_declared_outputs(script.parent, job.output_globs)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["VIBEQC_NO_HOSTNAME"] = "1"
    if "VQ_CPUS" in env:
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            env.setdefault(key, env["VQ_CPUS"])

    proc = subprocess.run(
        [sys.executable, script.name],
        cwd=script.parent,
        text=True,
        capture_output=True,
        env=env,
    )

    shutil.copy2(script, target / script.name)
    (target / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (target / "stderr.txt").write_text(proc.stderr, encoding="utf-8")

    copied: list[str] = []
    for pattern in job.output_globs:
        for path in sorted(script.parent.glob(pattern)):
            if path.is_file():
                shutil.copy2(path, target / path.name)
                copied.append(path.name)

    qvfs = sorted(name for name in copied if name.endswith(".qvf"))
    status = {
        "slug": job.slug,
        "script": str(script.relative_to(REPO)),
        "returncode": proc.returncode,
        "outputs": copied,
        "qvf_files": qvfs,
        "qvf_expected": job.qvf_expected,
    }
    (target / "artifact-status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "README.md").write_text(render_readme(job, status), encoding="utf-8")
    for path in target.iterdir():
        if path.is_file():
            sanitize_text_artifact(path)
    return status


def render_readme(job: Job, status: dict[str, object]) -> str:
    outputs = [str(x) for x in status.get("outputs", [])]
    qvfs = [o for o in outputs if o.endswith(".qvf")]
    script_rel = str(status.get("script", f"examples/**/{job.script}"))
    lines = [
        f"# {job.slug}",
        "",
        job.description,
        "",
        "Generated by `scripts/regenerate_doc_examples.py` from the canonical",
        f"input `{job.script}`.",
        "",
        "## Files",
        "",
        f"- `{job.script}`: input script",
        "- `stdout.txt`: full captured stdout from the queued run",
        "- `stderr.txt`: full captured stderr from the queued run",
        "- `artifact-status.json`: machine-readable run status",
    ]
    for name in outputs:
        lines.append(f"- `{name}`")
    if job.qvf_expected and not qvfs:
        lines.append("- QVF missing: expected from this input but not produced")
    if not job.qvf_expected:
        lines.append("- QVF not expected: this legacy diagnostic does not use the unified QVF writer")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```sh",
            f".venv/bin/python {script_rel}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def capture_screenshots(output_root: Path) -> list[dict[str, object]]:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "True")
    results: list[dict[str, object]] = []
    try:
        from vibeview.capture import capture_bands, capture_structure, capture_volume
        from vibeview.qvf import QVFReader
    except Exception as exc:
        marker = {"ok": False, "error": f"vibe-view import failed: {exc!r}"}
        (output_root / "screenshot-status.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return [marker]

    for qvf in sorted(output_root.glob("*/*.qvf")):
        bundle = qvf.parent
        record: dict[str, object] = {
            "qvf": str(qvf.relative_to(output_root)),
            "screenshots": [],
            "errors": [],
        }
        try:
            reader = QVFReader(qvf)
        except Exception as exc:
            record["errors"].append(f"open failed: {exc!r}")  # type: ignore[index]
            results.append(record)
            continue
        try:
            structure_out = bundle / f"{qvf.stem}-structure.png"
            if capture_structure(reader, structure_out):
                record["screenshots"].append(structure_out.name)  # type: ignore[index]

            density = next((s for s in reader.sections if s.kind == "volume.density"), None)
            if density is not None:
                density_out = bundle / f"{qvf.stem}-density.png"
                if capture_volume(reader, density.id, density_out, isovalue=0.05):
                    record["screenshots"].append(density_out.name)  # type: ignore[index]

            orbital = next(
                (s for s in reader.sections if s.kind in ("volume.orbital", "basis.ao")),
                None,
            )
            if orbital is not None:
                orbital_out = bundle / f"{qvf.stem}-orbital.png"
                if capture_volume(
                    reader,
                    orbital.id,
                    orbital_out,
                    isovalue=0.04,
                    colormap="coolwarm",
                ):
                    record["screenshots"].append(orbital_out.name)  # type: ignore[index]

            bands = next((s for s in reader.sections if s.kind == "bands"), None)
            if bands is not None:
                bands_out = bundle / f"{qvf.stem}-bands.png"
                if capture_bands(reader, bands_out, section_id=bands.id):
                    record["screenshots"].append(bands_out.name)  # type: ignore[index]
        except Exception as exc:
            record["errors"].append(f"capture failed: {exc!r}")  # type: ignore[index]
        finally:
            reader.close()

        (bundle / "screenshot-status.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(record)

    (output_root / "screenshot-status.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs(args.manifest)
    if args.only:
        wanted = set(args.only)
        known = {job.slug for job in jobs}
        missing = sorted(wanted - known)
        if missing:
            raise SystemExit(f"unknown slug(s): {', '.join(missing)}")
        jobs = [job for job in jobs if job.slug in wanted]

    statuses: list[dict[str, object]] = []
    if not args.capture_only:
        for job in jobs:
            print(f":: {job.slug} ({job.script})", flush=True)
            status = run_one(job, output_root)
            statuses.append(status)
            print(
                f"   rc={status['returncode']} files={len(status['outputs'])} "
                f"qvf={len(status['qvf_files'])}",
                flush=True,
            )
        (output_root / "artifact-index.json").write_text(
            json.dumps(statuses, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.screenshots:
        shots = capture_screenshots(output_root)
        print(f"screenshots: {len(shots)} qvf record(s)", flush=True)

    failures = [s for s in statuses if int(s.get("returncode", 1)) != 0]
    missing_qvf = [
        s
        for s in statuses
        if s.get("qvf_expected") and not s.get("qvf_files")
    ]
    if failures or missing_qvf:
        print("artifact generation completed with warnings", file=sys.stderr)
        for s in failures:
            print(f"  failed: {s['slug']} rc={s['returncode']}", file=sys.stderr)
        for s in missing_qvf:
            print(f"  missing qvf: {s['slug']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
