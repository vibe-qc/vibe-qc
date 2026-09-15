#!/usr/bin/env python3
"""Submit the Si diamond CRYSTAL23 job to compute-study via vq.

Produces: INPUT.d12 in the working directory, submits via vq,
polls for completion, fetches and prints results.

Usage:
    # Dry-run: validate D12 syntax only
    VIBEQC_DRY_RUN=1 .venv/bin/python submit_si_diamond_crystal.py

    # Submit to compute-study
    VIBEQC_HOST=compute-study .venv/bin/python submit_si_diamond_crystal.py

Environment variables:
    VIBEQC_HOST            vq host (default: from ~/.config/vq/config.toml)
    VIBEQC_CPUS            number of cores (default: 4)
    VIBEQC_WALL_TIME_SECONDS  wall time limit (default: 1800)

The script writes three D12 files (in INPUT.d12, INPUT_shrink8.d12,
INPUT_shrink2_no_mixing.d12).  The submission wrapper runs INPUT.d12.
After fetching, rename INPUT.d12 → INPUT_shrink8.d12 and re-submit
for the converged reference.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- D12 sources ---
D12_MAIN = HERE / "INPUT.d12"
D12_SHRINK8 = HERE / "INPUT_shrink8.d12"


def _find_vq() -> str:
    vq = shutil.which("vq")
    if vq is None:
        print(
            "ERROR: 'vq' not found on PATH. Install vibe-queue first.", file=sys.stderr
        )
        sys.exit(1)
    return vq


def _resolve_host(vq_bin: str) -> str:
    host = os.environ.get("VIBEQC_HOST", "")
    if host:
        return host
    # Try to read default_host from vq config
    cfg = Path.home() / ".config" / "vq" / "config.toml"
    if cfg.is_file():
        text = cfg.read_text()
        m = re.search(r'default_host\s*=\s*"(.*?)"', text)
        if m:
            return m.group(1)
    print(
        "ERROR: no host configured. Set VIBEQC_HOST=<host> or default_host in ~/.config/vq/config.toml",
        file=sys.stderr,
    )
    sys.exit(1)


def _find_wrapper() -> str:
    """Return path to run-crystal.sh on the remote host."""
    wrapper = os.environ.get("VIBEQC_CRYSTAL_WRAPPER", "")
    if wrapper:
        return wrapper
    # Default locations
    candidates = [
        "~/gitlab/vibe-queue/contrib/run-crystal.sh",
        "~/vibe-queue/contrib/run-crystal.sh",
    ]
    for c in candidates:
        p = Path(c).expanduser()
        if p.is_file():
            return str(p)
    print(
        "ERROR: CRYSTAL wrapper not found. Set VIBEQC_CRYSTAL_WRAPPER=<path>",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    vq_bin = _find_vq()
    host = _resolve_host(vq_bin)
    wrapper = _find_wrapper()
    cpus = int(os.environ.get("VIBEQC_CPUS", "4"))
    wall_s = int(os.environ.get("VIBEQC_WALL_TIME_SECONDS", "1800"))

    # --- Dry-run guard ---
    if os.environ.get("VIBEQC_DRY_RUN") == "1":
        print(f"[dry-run] Would submit to {host}:")
        print(
            f"  vq submit --host {host} -d {HERE} --cpus {cpus} "
            f"--wall-time-seconds {wall_s} --job-name si-diamond-rhf-sto3g-shrink2 "
            f"-- bash {wrapper} INPUT.d12 INPUT.out"
        )
        print()
        print(f"D12 files ready:")
        for f in sorted(HERE.glob("*.d12")):
            print(f"  {f.name}")
        return 0

    # --- Submit ---
    if not D12_MAIN.is_file():
        print(f"ERROR: {D12_MAIN} not found", file=sys.stderr)
        return 1

    job_name = "si-diamond-rhf-sto3g-shrink2"
    print(f"Submitting to {host}...")

    cmd = [
        vq_bin,
        "submit",
        "--host",
        host,
        "-d",
        str(HERE),
        "--cpus",
        str(cpus),
        "--wall-time-seconds",
        str(wall_s),
        "--job-name",
        job_name,
        "--",
        "bash",
        wrapper,
        "INPUT.d12",
        "INPUT.out",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(HERE),
    )
    if proc.returncode != 0:
        print(f"ERROR: vq submit failed (rc={proc.returncode})", file=sys.stderr)
        print(f"stderr: {proc.stderr[:500]}", file=sys.stderr)
        print(f"stdout: {proc.stdout[:500]}", file=sys.stderr)
        return 1

    # Parse jobid from vq stdout (first 12-hex token)
    jobid_match = re.search(r"[0-9a-f]{12}", proc.stdout)
    if not jobid_match:
        print(
            f"ERROR: could not parse jobid from vq output:\n{proc.stdout[:500]}",
            file=sys.stderr,
        )
        return 1
    jobid = jobid_match.group(0)
    print(f"Job submitted:  {jobid}")

    # --- Poll ---
    poll_s = float(os.environ.get("VIBEQC_POLL_INTERVAL_S", "15"))
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed > wall_s + 60:
            print(f"WARNING: wall-time exceeded, stopping poll")
            break

        st = subprocess.run(
            [vq_bin, "status", jobid, "--host", host],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = st.stdout + st.stderr
        if "completed" in output.lower() or "done" in output.lower():
            print(f"Job completed at {elapsed:.0f}s")
            break
        if "failed" in output.lower() or "error" in output.lower():
            print(f"Job failed:\n{output[:500]}")
            return 1

        time.sleep(poll_s)
        print(".", end="", flush=True)

    # --- Fetch ---
    print("\nFetching results...")
    fetch_dest = HERE / "fetched"
    fetch_dest.mkdir(exist_ok=True)
    subprocess.run(
        [
            vq_bin,
            "fetch",
            jobid,
            "--host",
            host,
            "-o",
            str(fetch_dest),
            "-d",
            str(HERE),
        ],
        capture_output=True,
        timeout=30,
    )

    out_path = fetch_dest / "INPUT.out"
    if out_path.is_file():
        text = out_path.read_text(errors="replace")
        # Extract total energy
        e_match = re.search(r"TOTAL ENERGY.*?=\s*([-+]?\d+\.\d+)", text, re.DOTALL)
        conv_match = re.search(r"SCF ENDED.*?CONVERGENCE", text, re.IGNORECASE)
        iter_match = re.search(r"SCF ENDED.*?(\d+)\s+CYCLES", text, re.DOTALL)
        if e_match:
            print(f"\nCRYSTAL23 TOTAL ENERGY = {e_match.group(1)} Ha")
        if conv_match:
            print("SCF converged: YES")
            if iter_match:
                print(f"SCF cycles:    {iter_match.group(1)}")
        else:
            print("SCF converged: NO (or could not parse)")
        # Print last 40 lines of output for SCF trace
        lines = text.splitlines()
        scf_start = -1
        for i, line in enumerate(lines):
            if "CYC" in line and "ETOT" in line:
                scf_start = i
        if scf_start >= 0:
            print("\nLast SCF cycles:")
            for line in lines[scf_start:]:
                print(line)
    else:
        print(f"WARNING: {out_path} not found after fetch")

    return 0


if __name__ == "__main__":
    sys.exit(main())
