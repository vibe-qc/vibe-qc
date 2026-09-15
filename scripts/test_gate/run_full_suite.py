#!/usr/bin/env python3
"""Per-file isolation triage runner for the vibe-qc suite.

For each test file or explicit pytest node id: spawn `pytest <target>` in its
own process, poll peak RSS of the whole process tree (psutil), enforce a hard
per-target wall-clock timeout (kill tree on exceed), and classify the outcome.
Writes one JSON line per target immediately to the output path (resumable: a
re-run skips targets already present).

Classification:
  PASS        rc==0
  FAIL        rc==1 (tests failed / errored, pytest exited cleanly)
  SEGFAULT    rc==-11 (SIGSEGV) — a real crash bug
  ABORT       rc==-6  (SIGABRT)
  OOM_KILLED  rc==-9  (SIGKILL), NOT our timeout, and the sampled peak RSS of
              the process tree reached --oom-fraction of physical memory
              (default 0.5): an OOM kill with evidence behind it
  SIGKILLED   rc==-9  (SIGKILL), NOT our timeout, and no such memory evidence:
              cause unassigned (an outside kill, a supervisor, a short spike
              the 250 ms sampler missed); never call it OOM (#218)
  TIMEOUT     our per-file wall-clock killed it (possible infinite loop)
  NOTESTS     rc==5 (no tests collected)
  COLLECT_ERR rc in (2,3,4) (usage/collection/internal error)
  OTHER_SIGNAL rc<0 other

Usage:
  python vqc_triage_runner.py --wt <worktree> [--out PATH] [--jobs N]
         [--file-timeout S] [--test-timeout S] [--only test_a.py,test_b.py]
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None

SUMMARY_RE = re.compile(
    r"(\d+) (passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warning|warnings)"
)
DEFAULT_JOBS = 6
DEFAULT_FILE_TIMEOUT = 1200.0
DEFAULT_TEST_TIMEOUT = 600
DEFAULT_LANE_MANIFEST = Path(__file__).with_name("lane_manifest.json")
DEFAULT_SUITE_MANIFEST = Path(__file__).with_name("suite_manifest.json")
GATE_TIERS = {"T0", "T1", "T2", "T3"}
METHOD_MATURITY_STATES = {"production", "verified", "under-review", "experimental"}
LANE_CLASSES = {"pre-cut blocking", "post-cut evaluation", "implementing-chat-only"}
# A release line ("v0.18.x") or a forward track ("v2.0"). Never a single patch.
TARGET_RELEASE_RE = re.compile(r"^v(\d+)\.(\d+)(?:\.x)?$")
REQUIRED_LANE_METADATA = (
    "owner",
    "method_maturity",
    "lane_class",
    "global_items",
    "target_release",
    "scientific_acceptance",
    "required_full_calculation",
)


def parse_counts(text: str) -> dict:
    counts = {}
    # last summary line wins
    for m in SUMMARY_RE.finditer(text):
        key = m.group(2).rstrip("s")  # normalise plural
        counts[key] = int(m.group(1))
    return counts


def peak_rss_mb(proc: "subprocess.Popen", interval: float, deadline: float) -> tuple:
    """Poll RSS of the process tree until it exits or deadline. Returns
    (peak_mb, timed_out). Kills the tree on deadline."""
    peak = 0
    timed_out = False
    if psutil is None:
        # fall back: just wait with timeout, no RSS
        try:
            proc.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc.pid)
        return (0, timed_out)
    try:
        p = psutil.Process(proc.pid)
    except Exception:  # noqa: BLE001
        return (0, False)
    while True:
        if proc.poll() is not None:
            break
        if time.time() > deadline:
            timed_out = True
            _kill_tree(proc.pid)
            try:
                proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                pass
            break
        rss = 0
        try:
            procs = [p] + p.children(recursive=True)
            for q in procs:
                try:
                    rss += q.memory_info().rss
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        if rss > peak:
            peak = rss
        time.sleep(interval)
    return (round(peak / 1e6, 1), timed_out)


def _kill_tree(pid: int) -> None:
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            for c in p.children(recursive=True):
                try:
                    c.kill()
                except Exception:  # noqa: BLE001
                    pass
            p.kill()
            return
        except Exception:  # noqa: BLE001
            pass
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


DEFAULT_OOM_FRACTION = 0.5


def physical_memory_mb() -> float | None:
    """Total physical memory in MB, or None when psutil is unavailable."""
    if psutil is None:
        return None
    try:
        return round(psutil.virtual_memory().total / 1e6, 1)
    except Exception:  # noqa: BLE001
        return None


def oom_evidence(peak_rss_mb, memory_total_mb, oom_fraction=DEFAULT_OOM_FRACTION) -> dict:
    """Whether a SIGKILL can honestly be called an OOM kill.

    The only memory evidence the runner has is the 250 ms RSS sample of the
    process tree. A kill is reported as OOM only when that peak reached
    ``oom_fraction`` of physical memory; a 387 MB peak on a 128 GB box is not
    an OOM story (#218). The returned dict is stored on the row so a reader
    can see the numbers the label rests on.
    """
    peak = float(peak_rss_mb or 0.0)
    total = float(memory_total_mb) if memory_total_mb else None
    fraction = (peak / total) if total else None
    return {
        "peak_rss_mb": peak,
        "memory_total_mb": total,
        "peak_fraction_of_memory": None if fraction is None else round(fraction, 4),
        "oom_fraction_threshold": float(oom_fraction),
        "oom_established": bool(fraction is not None and fraction >= float(oom_fraction)),
    }


def classify(rc, timed_out, counts, evidence=None) -> str:
    """Map a target's exit to a status label.

    ``evidence`` is the :func:`oom_evidence` dict; without it a SIGKILL is
    ``SIGKILLED`` (cause unassigned), never ``OOM_KILLED``.
    """
    if timed_out:
        return "TIMEOUT"
    if rc == 0:
        return "PASS"
    if rc == 1:
        return "FAIL"
    if rc == 5:
        return "NOTESTS"
    if rc in (2, 3, 4):
        return "COLLECT_ERR"
    if rc is not None and rc < 0:
        sig = -rc
        if sig == 9:
            established = bool(evidence and evidence.get("oom_established"))
            return "OOM_KILLED" if established else "SIGKILLED"
        return {11: "SEGFAULT", 6: "ABORT"}.get(sig, f"SIGNAL_{sig}")
    return f"RC_{rc}"


def _read_lane_manifest(path: Path) -> dict:
    with path.open() as fh:
        data = json.load(fh)
    lanes = data.get("lanes")
    if not isinstance(lanes, dict):
        raise ValueError(f"{path}: missing object field 'lanes'")
    validate_lane_manifest(data, path)
    return data


def load_lane_manifest(path: Path) -> dict:
    """Load the optional gate-lane manifest."""
    data = _read_lane_manifest(path)
    _warn_stale_target_releases(data, path)
    return data


def load_suite_manifest(path: Path) -> dict:
    """Load and validate the exhaustive per-file tier manifest."""
    with path.open() as fh:
        data = json.load(fh)
    rows = data.get("tests")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing list field 'tests'")
    seen = set()
    for row in rows:
        rel = row.get("file")
        if not rel or rel in seen:
            raise ValueError(f"{path}: duplicate or missing test file {rel!r}")
        seen.add(rel)
        if row.get("tier") not in GATE_TIERS:
            raise ValueError(f"{path}: {rel} has invalid tier {row.get('tier')!r}")
        if row.get("maturity") not in METHOD_MATURITY_STATES:
            raise ValueError(
                f"{path}: {rel} has invalid maturity {row.get('maturity')!r}"
            )
        if row["maturity"] == "experimental" and row["tier"] != "T3":
            raise ValueError(f"{path}: experimental file {rel} must be T3")
        if row["tier"] in {"T0", "T1"} and not row.get("owner"):
            raise ValueError(f"{path}: blocking file {rel} has no owner")
    return data


def parse_tiers(value: str) -> list[str]:
    tiers = _dedupe(part.strip().upper() for part in value.split(",") if part.strip())
    invalid = sorted(set(tiers) - GATE_TIERS)
    if invalid:
        raise ValueError(f"unknown tier(s): {', '.join(invalid)}")
    return tiers


def select_tier_targets(suite_manifest: dict, tiers: list[str]) -> list[str]:
    selected = set(tiers)
    return [row["file"] for row in suite_manifest["tests"] if row["tier"] in selected]


def validate_lane_manifest(data: dict, path: Path) -> None:
    """Validate release-policy metadata in the lane manifest."""
    policy = data.get("policy") or {}
    configured_states = set(policy.get("method_maturity_states") or [])
    if configured_states != METHOD_MATURITY_STATES:
        raise ValueError(
            f"{path}: policy.method_maturity_states must be "
            f"{sorted(METHOD_MATURITY_STATES)}"
        )
    configured_classes = set(policy.get("lane_classes") or [])
    if configured_classes != LANE_CLASSES:
        raise ValueError(f"{path}: policy.lane_classes must be {sorted(LANE_CLASSES)}")
    acceptance = policy.get("scientific_acceptance") or {}
    if acceptance.get("pytest_or_ci_can_accept_method") is not False:
        raise ValueError(f"{path}: pytest/CI must not be a scientific acceptance gate")

    lanes = data["lanes"]
    for name, lane in lanes.items():
        missing = [field for field in REQUIRED_LANE_METADATA if field not in lane]
        if missing:
            raise ValueError(f"{path}: lane {name!r} missing metadata: {missing}")
        maturity = lane["method_maturity"]
        if maturity not in METHOD_MATURITY_STATES:
            raise ValueError(f"{path}: lane {name!r} has invalid maturity {maturity!r}")
        lane_class = lane["lane_class"]
        if lane_class not in LANE_CLASSES:
            raise ValueError(f"{path}: lane {name!r} has invalid lane_class {lane_class!r}")
        if lane["scientific_acceptance"] is not False:
            raise ValueError(f"{path}: lane {name!r} cannot accept scientific values")
        target_release = lane.get("target_release")
        if not target_release:
            raise ValueError(f"{path}: lane {name!r} must name a target_release")
        if not TARGET_RELEASE_RE.match(str(target_release)):
            raise ValueError(
                f"{path}: lane {name!r} has invalid target_release {target_release!r}; "
                "expected a release line such as 'v0.18.x' or a forward track such as 'v2.0'"
            )
        if not lane.get("required_full_calculation"):
            raise ValueError(
                f"{path}: lane {name!r} must describe required_full_calculation"
            )
        global_items = lane.get("global_items")
        if not isinstance(global_items, list) or not global_items:
            raise ValueError(f"{path}: lane {name!r} must link global_items")
        if lane_class == "implementing-chat-only":
            if maturity != "experimental":
                raise ValueError(
                    f"{path}: lane {name!r} is implementing-chat-only but not experimental"
                )
            if lane.get("impact_default") is not False:
                raise ValueError(
                    f"{path}: experimental lane {name!r} must opt out of impact gates"
                )
            if lane.get("release_impact_default") is True:
                raise ValueError(
                    f"{path}: experimental lane {name!r} cannot opt into release impact"
                )
        elif maturity == "experimental":
            raise ValueError(
                f"{path}: experimental lane {name!r} must be implementing-chat-only"
            )

    for profile_name, profile in (data.get("profiles") or {}).items():
        for lane_name in profile.get("blocking_lanes") or []:
            if lane_name not in lanes:
                raise ValueError(
                    f"{path}: profile {profile_name!r} references unknown "
                    f"blocking lane {lane_name!r}"
                )
            lane = lanes[lane_name]
            if lane["lane_class"] != "pre-cut blocking":
                raise ValueError(
                    f"{path}: profile {profile_name!r} blocking lane {lane_name!r} "
                    "must be pre-cut blocking"
                )
        for lane_name in profile.get("advisory_lanes") or []:
            if lane_name not in lanes:
                raise ValueError(
                    f"{path}: profile {profile_name!r} references unknown "
                    f"advisory lane {lane_name!r}"
                )
            lane = lanes[lane_name]
            if lane["lane_class"] == "implementing-chat-only":
                raise ValueError(
                    f"{path}: profile {profile_name!r} cannot include experimental "
                    f"lane {lane_name!r}"
                )


def current_release_line(pyproject: Path) -> tuple[int, int] | None:
    """Return the (major, minor) line of ``[project] version`` in *pyproject*.

    ``target_release`` staleness is judged against this. It is read rather than
    declared, so the check cannot rot the way a hand-maintained "current line"
    constant would. Returns ``None`` when the file or version is unavailable.
    """
    try:
        with pyproject.open("rb") as fh:
            version = tomllib.load(fh)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None
    match = re.match(r"(\d+)\.(\d+)", str(version))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _manifest_pyproject(path: Path) -> Path:
    """The pyproject.toml of the repository a lane manifest belongs to."""
    resolved = path.resolve()
    try:
        return resolved.parents[2] / "pyproject.toml"
    except IndexError:
        return resolved.with_name("pyproject.toml")


def stale_target_release_report(manifest: dict, current_line: tuple[int, int]) -> dict:
    """Classify ``target_release`` staleness against *current_line*.

    A lane is stale when its release line is older than the repository's current
    line; forward tracks such as ``v2.0`` are never stale. Stale lanes listed in
    ``policy.known_stale_target_release.lanes`` are tolerated, mirroring
    ``known_reds_baseline.json``: only an untracked stale lane, a tracked lane
    that is no longer stale, or a tracked name that is not a lane needs action.
    """
    policy = manifest.get("policy") or {}
    tracked = set((policy.get("known_stale_target_release") or {}).get("lanes") or [])
    lanes = manifest["lanes"]
    stale = set()
    for name, lane in lanes.items():
        match = TARGET_RELEASE_RE.match(str(lane.get("target_release", "")))
        if match and (int(match.group(1)), int(match.group(2))) < current_line:
            stale.add(name)
    return {
        "current_line": current_line,
        "stale": sorted(stale),
        "tracked": sorted(tracked & stale),
        "untracked": sorted(stale - tracked),
        "no_longer_stale": sorted((tracked & set(lanes)) - stale),
        "unknown": sorted(tracked - set(lanes)),
    }


def _stale_target_release_problems(report: dict) -> list[str]:
    line = "v{}.{}".format(*report["current_line"])
    tracking = "policy.known_stale_target_release"
    problems = [
        f"lane {name!r} targets a release line older than {line}; "
        f"its owning chat retargets it, or it is tracked in {tracking}"
        for name in report["untracked"]
    ]
    problems += [
        f"lane {name!r} is no longer stale; remove it from {tracking}"
        for name in report["no_longer_stale"]
    ]
    problems += [
        f"{tracking} names unknown lane {name!r}; remove it"
        for name in report["unknown"]
    ]
    return problems


def _warn_stale_target_releases(data: dict, path: Path) -> None:
    """Report ``target_release`` staleness on stderr without failing the run.

    Loading the manifest is the everyday lane workflow, and every version bump
    makes the previous line stale at once, so failing here would break unrelated
    lane runs. ``check_lane_manifest`` and the contract test enforce; this only
    keeps the signal visible.
    """
    current_line = current_release_line(_manifest_pyproject(path))
    if current_line is None:
        return
    for problem in _stale_target_release_problems(
        stale_target_release_report(data, current_line)
    ):
        print(f"[triage] warning: {path.name}: {problem}", file=sys.stderr)


def check_lane_manifest(path: Path) -> int:
    """Validate the lane manifest, including ``target_release`` staleness.

    Returns a process exit code: 0 when every stale lane is tracked, 1 when
    action is needed or the current release line cannot be read. Nothing here
    imports vibeqc, so this is the form of the check any CI job can enforce.
    """
    try:
        data = _read_lane_manifest(path)
    except (OSError, ValueError) as exc:
        print(f"[triage] lane manifest invalid: {exc}", file=sys.stderr)
        return 1
    pyproject = _manifest_pyproject(path)
    current_line = current_release_line(pyproject)
    if current_line is None:
        print(f"[triage] cannot read the current release line from {pyproject}", file=sys.stderr)
        return 1
    report = stale_target_release_report(data, current_line)
    ref = ((data.get("policy") or {}).get("known_stale_target_release") or {}).get("ref")
    tracked_note = f" ({ref})" if ref else ""
    print(
        "[triage] target_release checked against v{}.{}: ".format(*current_line)
        + f"{len(report['stale'])} stale, {len(report['tracked'])} tracked{tracked_note}"
    )
    problems = _stale_target_release_problems(report)
    for problem in problems:
        print(f"[triage] {problem}", file=sys.stderr)
    return 1 if problems else 0


def profile_lane_names(manifest: dict, profile_name: str, mode: str) -> list[str]:
    """Return lane names from a release/test profile."""
    profiles = manifest.get("profiles") or {}
    try:
        profile = profiles[profile_name]
    except KeyError as exc:
        names = ", ".join(sorted(profiles)) or "(none)"
        raise ValueError(f"unknown profile {profile_name!r}; available profiles: {names}") from exc
    blocking = list(profile.get("blocking_lanes") or [])
    advisory = list(profile.get("advisory_lanes") or [])
    if mode == "blocking":
        return _dedupe(blocking)
    if mode == "advisory":
        return _dedupe(advisory)
    if mode == "all":
        return _dedupe(blocking + advisory)
    raise ValueError(f"unknown profile mode {mode!r}")


def _matches_any(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(rel, pat) for pat in patterns)


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_lane_names(raw: str) -> list[str]:
    """Parse a comma-separated lane list."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def select_lane_files(files: list[Path], wt: Path, lane: dict) -> list[Path]:
    """Return files selected by a lane's include/exclude glob rules."""
    include = (
        ["tests/test_*.py"]
        if lane.get("include") is None
        else list(lane.get("include") or [])
    )
    exclude = list(lane.get("exclude") or [])
    selected = []
    for f in files:
        rel = f.relative_to(wt).as_posix()
        if _matches_any(rel, include) and not _matches_any(rel, exclude):
            selected.append(f)
    return selected


def select_lane_set_files(files: list[Path], wt: Path, lanes: list[dict]) -> list[Path]:
    """Return the union of files selected by multiple lanes."""
    selected = set()
    for lane in lanes:
        selected.update(select_lane_files(files, wt, lane))
    return [f for f in files if f in selected]


def _node_file_part(target: str) -> str:
    """Return the file portion of a pytest target."""
    return target.split("::", 1)[0]


def select_lane_targets(files: list[Path], wt: Path, lane: dict) -> list[str]:
    """Return file paths and explicit pytest node ids selected by a lane."""
    targets = [
        f.relative_to(wt).as_posix()
        for f in select_lane_files(files, wt, lane)
    ]
    targets.extend(str(node) for node in lane.get("nodes") or [])
    return _dedupe(targets)


def select_lane_set_targets(files: list[Path], wt: Path, lanes: list[dict]) -> list[str]:
    """Return the union of test targets selected by multiple lanes.

    File targets keep repository discovery order. Explicit node ids keep lane
    order and are skipped when their full file is already selected.
    """
    selected_files: set[str] = set()
    explicit_nodes: list[str] = []
    for lane in lanes:
        selected_files.update(
            f.relative_to(wt).as_posix() for f in select_lane_files(files, wt, lane)
        )
        explicit_nodes.extend(str(node) for node in lane.get("nodes") or [])

    targets = [
        f.relative_to(wt).as_posix()
        for f in files
        if f.relative_to(wt).as_posix() in selected_files
    ]
    targets.extend(
        node
        for node in explicit_nodes
        if _node_file_part(node) not in selected_files
    )
    return _dedupe(targets)


def combined_lane_defaults(lanes: list[dict], explicit_markexpr: bool = False) -> dict | None:
    """Merge runtime defaults for a multi-lane run.

    The union runner launches every selected file with the same runtime knobs,
    so use conservative defaults: lowest concurrency and highest timeout.
    """
    if not lanes:
        return None
    markexprs = {str(lane["markexpr"]) for lane in lanes if lane.get("markexpr")}
    if len(markexprs) > 1 and not explicit_markexpr:
        labels = ", ".join(sorted(markexprs))
        raise ValueError(
            "cannot combine lanes with different markexpr defaults "
            f"({labels}); run them separately or pass --markexpr explicitly"
        )
    jobs = [int(lane["jobs"]) for lane in lanes if lane.get("jobs") is not None]
    file_timeouts = [
        float(lane["file_timeout_s"])
        for lane in lanes
        if lane.get("file_timeout_s") is not None
    ]
    test_timeouts = [
        int(lane["test_timeout_s"])
        for lane in lanes
        if lane.get("test_timeout_s") is not None
    ]
    merged = {
        "markexpr": next(iter(markexprs)) if len(markexprs) == 1 else "",
        "jobs": min(jobs) if jobs else None,
        "file_timeout_s": max(file_timeouts) if file_timeouts else None,
        "test_timeout_s": max(test_timeouts) if test_timeouts else None,
        "heavy_env": any(bool(lane.get("heavy_env")) for lane in lanes),
    }
    return merged


def git_changed_paths(wt: Path, ref: str) -> list[str]:
    """Return repo-relative paths changed between ref and HEAD."""
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}...HEAD"],
        cwd=str(wt),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"git diff failed for --changed-since {ref!r}:\n{proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def affected_lane_names(
    manifest: dict,
    changed_paths: list[str],
    impact_mode: str = "dev",
) -> list[str]:
    """Infer non-aggregate lanes whose trigger paths or test globs changed."""
    if impact_mode not in {"dev", "release"}:
        raise ValueError(f"unknown impact mode {impact_mode!r}")
    out = []
    for name, lane in manifest["lanes"].items():
        if lane.get("impact_default") is False:
            continue
        if impact_mode == "release" and lane.get("release_impact_default") is not True:
            continue
        triggers = list(lane.get("trigger_paths") or [])
        include = list(lane.get("include") or [])
        exclude = list(lane.get("exclude") or [])
        node_files = {
            _node_file_part(str(node))
            for node in lane.get("nodes") or []
        }
        for rel in changed_paths:
            lane_test_changed = (
                (
                    _matches_any(rel, include)
                    and not _matches_any(rel, exclude)
                )
                or rel in node_files
            )
            if _matches_any(rel, triggers) or lane_test_changed:
                out.append(name)
                break
    return _dedupe(out)


def print_affected_lanes(ref: str, changed_paths: list[str], lanes: list[str]) -> None:
    print(f"[triage] changed since {ref}: {len(changed_paths)} paths")
    for rel in changed_paths:
        print(f"  {rel}")
    print("[triage] affected lanes: " + (", ".join(lanes) if lanes else "(none)"))


def apply_lane_defaults(args, lane: dict | None):
    """Fill unset runtime knobs from a lane record, then global defaults."""
    if lane:
        if not args.markexpr and lane.get("markexpr"):
            args.markexpr = str(lane["markexpr"])
        if args.jobs is None and lane.get("jobs") is not None:
            args.jobs = int(lane["jobs"])
        if args.file_timeout is None and lane.get("file_timeout_s") is not None:
            args.file_timeout = float(lane["file_timeout_s"])
        if args.test_timeout is None and lane.get("test_timeout_s") is not None:
            args.test_timeout = int(lane["test_timeout_s"])
        if bool(lane.get("heavy_env")):
            args.heavy_env = True

    if args.jobs is None:
        args.jobs = DEFAULT_JOBS
    if args.file_timeout is None:
        args.file_timeout = DEFAULT_FILE_TIMEOUT
    if args.test_timeout is None:
        args.test_timeout = DEFAULT_TEST_TIMEOUT
    return args


def print_lanes(manifest: dict) -> None:
    print("Available test-gate lanes:")
    for name, lane in sorted(manifest["lanes"].items()):
        maturity = lane.get("method_maturity", "")
        lane_class = lane.get("lane_class", "")
        cadence = lane.get("cadence", "")
        owner = lane.get("owner", "")
        desc = lane.get("description", "")
        print(
            f"  {name:28s} {maturity:13s} {lane_class:22s} "
            f"{cadence:16s} {owner:18s} {desc}"
        )


def print_profiles(manifest: dict) -> None:
    print("Available test-gate profiles:")
    for name, profile in sorted((manifest.get("profiles") or {}).items()):
        desc = profile.get("description", "")
        blocking = ", ".join(profile.get("blocking_lanes") or [])
        advisory = ", ".join(profile.get("advisory_lanes") or [])
        print(f"  {name:28s} {desc}")
        print(f"    blocking: {blocking or '(none)'}")
        print(f"    advisory: {advisory or '(none)'}")


def discover_test_files(wt: Path) -> list[Path]:
    """Return the test files known to the per-file gate runner."""
    files = sorted((wt / "tests").glob("test_*.py"))
    files += sorted((wt / "tests" / "basisset_dev").glob("test_*.py"))
    return files


def print_selected_files(files: list[Path], wt: Path, lane_name: str = "") -> None:
    targets = [f.relative_to(wt).as_posix() for f in files]
    print_selected_targets(targets, lane_name)


def print_selected_targets(targets: list[str], lane_name: str = "") -> None:
    lane_note = f" lane={lane_name}" if lane_name else ""
    print(f"[triage]{lane_note} dry-run selected {len(targets)} targets")
    for target in targets:
        print(f"  {target}")


def _resolved_test_target_file(wt: Path, target: str) -> Path:
    """Resolve a pytest node's file beneath the selected worktree."""
    raw = _node_file_part(target).replace("\\", "/")
    candidate = Path(raw)
    resolved = (
        candidate.resolve(strict=False)
        if candidate.is_absolute()
        else (wt / candidate).resolve(strict=False)
    )
    worktree = wt.resolve(strict=False)
    try:
        resolved.relative_to(worktree)
    except ValueError as exc:
        raise ValueError(
            f"refusing test target outside worktree {worktree}: {resolved}"
        ) from exc
    return resolved


@contextlib.contextmanager
def _isolated_test_environment(wt: Path, target: str) -> Iterator[dict[str, str]]:
    """Give each vq pytest child no route to caller-owned persistent state."""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    target_file = _resolved_test_target_file(wt, target)
    vq_root = (wt / "vibe-queue").resolve(strict=False)
    try:
        target_file.relative_to(vq_root)
    except ValueError:
        # #527 concerns vq persistence. Preserve the established environment
        # contract for every unrelated core/basis/view test-gate target.
        yield env
        return

    # Keep the platform's caller-owned temporary parent. On macOS, forcing
    # /tmp gives the sandbox group ``wheel`` and strips setgid bits in the
    # multi-user permission tests. The vq conftest supplies its own short
    # /tmp symlink for AF_UNIX sockets, so the outer path need not be short.
    with tempfile.TemporaryDirectory(
        prefix="vibeqc-test-gate-",
    ) as sandbox_name:
        sandbox = Path(sandbox_name)
        roots = {
            "HOME": sandbox / "home",
            "TMPDIR": sandbox / "tmp",
            "XDG_DATA_HOME": sandbox / "xdg-data",
            "XDG_CONFIG_HOME": sandbox / "xdg-config",
            "XDG_CACHE_HOME": sandbox / "xdg-cache",
            "VQ_STATE_DIR": sandbox / "vq-state",
            "VQ_CONFIG_DIR": sandbox / "vq-config",
            "VQ_ARCHIVE_DIR": sandbox / "vq-archive",
            "VQ_MULTI_USER_ROOT": sandbox / "vq-multi-user",
            "VQ_TEST_SANDBOX_ROOT": sandbox,
            "PYTHONPYCACHEPREFIX": sandbox / "pycache",
        }
        for path in roots.values():
            path.mkdir(parents=True, exist_ok=True)

        # Command-line options are assembled below. An inherited
        # PYTEST_ADDOPTS=--noconftest bypassed vq's isolation fixture in #527;
        # -o addopts= does not neutralize that environment variable.
        env.pop("PYTEST_ADDOPTS", None)
        env.pop("PYTEST_CURRENT_TEST", None)
        env.pop("PYTEST_VERSION", None)
        env.update({name: str(path) for name, path in roots.items()})
        env["VQ_TEST_SHORT_TMPDIR"] = str(roots["TMPDIR"])
        env["VQ_TEST_SYSTEM_CONFIG_FILE"] = str(
            roots["VQ_CONFIG_DIR"] / "system-config.toml"
        )
        env["VQ_WEB_TOKEN_FILE"] = str(roots["VQ_CONFIG_DIR"] / "web-token")
        source_root = str((wt / "vibe-queue" / "src").resolve())
        inherited = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            source_root if not inherited
            else source_root + os.pathsep + inherited
        )
        yield env


def run_one(tf: Path | str, wt: Path, py: str, file_timeout: float, test_timeout: int,
            markexpr: str = "", heavy_env: bool = False,
            oom_fraction: float = DEFAULT_OOM_FRACTION) -> dict:
    if isinstance(tf, Path):
        rel = str(tf.relative_to(wt))
    else:
        rel = str(tf)
    # ``-v`` rather than ``-q``: pytest prints the node id before a test runs
    # and appends the outcome after it, so when the child is killed mid-test
    # the last line of the tail names the node that was running. With ``-q``
    # a killed file left nothing but progress dots (#218).
    cmd = [
        py, "-m", "pytest", rel,
        "-p", "no:cacheprovider", "-v", "--no-header",
        "-o", "addopts=", "-o", "console_output_style=classic",
        f"--timeout={test_timeout}", "--timeout-method=thread",
    ]
    if markexpr:
        cmd += ["-m", markexpr]
    with _isolated_test_environment(wt, rel) as isolated_env:
        if heavy_env:
            # The env-gated heavy tests opt in via this var so the heavy/slow
            # lane executes them rather than reporting skips.
            isolated_env["VIBEQC_RUN_HEAVY_TESTS"] = "1"
        t0 = time.time()
        proc = subprocess.Popen(
            cmd, cwd=str(wt), env=isolated_env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # drain output in a thread while polling RSS
        out_chunks = []

        import threading

        def _drain():
            try:
                for line in proc.stdout:
                    out_chunks.append(line)
            except Exception:  # noqa: BLE001
                pass

        th = threading.Thread(target=_drain, daemon=True)
        th.start()
        peak, timed_out = peak_rss_mb(proc, 0.25, t0 + file_timeout)
        rc = proc.poll()
        th.join(timeout=5)
        dt = round(time.time() - t0, 1)
        text = "".join(out_chunks)
        counts = parse_counts(text)
        evidence = oom_evidence(peak, physical_memory_mb(), oom_fraction)
        status = classify(rc, timed_out, counts, evidence)
        # capture a useful tail
        tail = "\n".join(text.splitlines()[-25:])[-3000:]
        record = {
            "file": rel,
            "status": status,
            "rc": rc,
            "elapsed_s": dt,
            "peak_rss_mb": peak,
            "counts": counts,
            "timed_out": timed_out,
            "tail": tail,
        }
        if rc is not None and rc < 0:
            record["signal"] = -rc
            record["child_pid"] = proc.pid
            if -rc == 9:
                record["oom_evidence"] = evidence
        return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt", required=True)
    ap.add_argument("--out", default=os.environ.get("VQ_WORKDIR", ".") + "/triage.jsonl")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--file-timeout", type=float, default=None)
    ap.add_argument("--test-timeout", type=int, default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--markexpr", default="", help="pytest -m expression, e.g. 'not slow' or 'slow'")
    ap.add_argument("--heavy-env", action="store_true", help="set VIBEQC_RUN_HEAVY_TESTS=1")
    ap.add_argument(
        "--oom-fraction", type=float, default=DEFAULT_OOM_FRACTION,
        help="a SIGKILL is reported as OOM_KILLED only if the sampled peak RSS of the "
             "process tree reached this fraction of physical memory; otherwise SIGKILLED "
             f"(cause unassigned). Default {DEFAULT_OOM_FRACTION}.",
    )
    ap.add_argument("--lane", default="", help="named lane(s), comma-separated, from scripts/test_gate/lane_manifest.json")
    ap.add_argument("--profile", default="", help="named profile from scripts/test_gate/lane_manifest.json")
    ap.add_argument(
        "--profile-mode",
        choices=("blocking", "advisory", "all"),
        default="blocking",
        help="which part of --profile to select",
    )
    ap.add_argument("--lane-manifest", default=str(DEFAULT_LANE_MANIFEST))
    ap.add_argument("--suite-manifest", default=str(DEFAULT_SUITE_MANIFEST))
    ap.add_argument(
        "--tier",
        default="",
        help="gate tier(s), comma-separated: T0,T1,T2,T3",
    )
    ap.add_argument("--list-lanes", action="store_true", help="print lane names and exit")
    ap.add_argument("--list-profiles", action="store_true", help="print profile names and exit")
    ap.add_argument("--changed-since", default="", help="infer affected lanes from git diff REF...HEAD")
    ap.add_argument(
        "--impact-mode",
        choices=("dev", "release"),
        default="dev",
        help=(
            "lane inference mode for --changed-since: dev selects full area "
            "lanes; release selects cheap release-impact lanes only"
        ),
    )
    ap.add_argument("--list-affected-lanes", action="store_true", help="print lanes inferred by --changed-since and exit")
    ap.add_argument(
        "--check-lane-manifest",
        action="store_true",
        help="validate lane_manifest.json, including target_release staleness, and exit non-zero on problems",
    )
    ap.add_argument("--dry-run", action="store_true", help="print selected test targets and exit")
    args = ap.parse_args()

    wt = Path(args.wt).resolve()
    suite_manifest_path = Path(args.suite_manifest)
    if not suite_manifest_path.is_absolute():
        suite_manifest_path = (wt / suite_manifest_path).resolve()
    try:
        suite_manifest = load_suite_manifest(suite_manifest_path)
        tier_names = parse_tiers(args.tier)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.tier and (args.lane or args.profile or args.changed_since):
        raise SystemExit("--tier cannot be combined with lane/profile/impact selection")
    lane_names = []
    lanes = []
    manifest = None
    if (
        args.lane
        or args.profile
        or args.list_lanes
        or args.list_profiles
        or args.changed_since
        or args.list_affected_lanes
        or args.check_lane_manifest
    ):
        manifest_path = Path(args.lane_manifest)
        if not manifest_path.is_absolute():
            manifest_path = (wt / manifest_path).resolve()
        if args.check_lane_manifest:
            raise SystemExit(check_lane_manifest(manifest_path))
        manifest = load_lane_manifest(manifest_path)
        if args.list_lanes:
            print_lanes(manifest)
            return
        if args.list_profiles:
            print_profiles(manifest)
            return
        affected = []
        changed_paths = []
        if args.changed_since:
            changed_paths = git_changed_paths(wt, args.changed_since)
            affected = affected_lane_names(manifest, changed_paths, args.impact_mode)
            affected = _dedupe(["smoke"] + affected)
        elif args.list_affected_lanes:
            raise SystemExit("--list-affected-lanes requires --changed-since REF")
        if args.list_affected_lanes:
            print_affected_lanes(args.changed_since, changed_paths, affected)
            return
        profile_lanes = []
        if args.profile:
            try:
                profile_lanes = profile_lane_names(
                    manifest,
                    args.profile,
                    args.profile_mode,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        lane_names = _dedupe(profile_lanes + parse_lane_names(args.lane) + affected)
        for lane_name in lane_names:
            try:
                lanes.append(manifest["lanes"][lane_name])
            except KeyError:
                names = ", ".join(sorted(manifest["lanes"]))
                raise SystemExit(f"unknown lane {lane_name!r}; available lanes: {names}")

    if tier_names and args.jobs is None:
        # Blocking T1 mixes medium-memory numerical sentinels.  Two workers
        # stays inside the documented memory envelope and avoids the six-way
        # starvation that previously triggered serial re-verification loops.
        if tier_names == ["T0"]:
            args.jobs = 6
        elif set(tier_names) <= {"T0", "T1"}:
            args.jobs = 2
        else:
            # T2/T3 contain the documented XL files.  Keep their default
            # shard serial; callers may explicitly raise concurrency on a
            # host whose memory routing is independently controlled.
            args.jobs = 1
    if tier_names and set(tier_names) & {"T2", "T3"}:
        if args.file_timeout is None:
            args.file_timeout = 2400.0
        if args.test_timeout is None:
            args.test_timeout = 1800
        args.heavy_env = True
    lane_defaults = combined_lane_defaults(lanes, explicit_markexpr=bool(args.markexpr))
    args = apply_lane_defaults(args, lane_defaults)
    files = discover_test_files(wt)
    targets = [f.relative_to(wt).as_posix() for f in files]
    if tier_names:
        targets = select_tier_targets(suite_manifest, tier_names)
    if lanes:
        targets = select_lane_set_targets(files, wt, lanes)
    if args.only:
        wanted = set(args.only.split(","))
        targets = [
            target for target in targets
            if (
                target in wanted
                or _node_file_part(target) in wanted
                or Path(_node_file_part(target)).name in wanted
            )
        ]
    if args.dry_run:
        print_selected_targets(targets, ",".join(tier_names or lane_names))
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["file"])
            except Exception:  # noqa: BLE001
                pass
    todo = [target for target in targets if target not in done]
    lane_note = f" lane={','.join(lane_names)}" if lane_names else ""
    if tier_names:
        lane_note = f" tier={','.join(tier_names)}"
    suite_by_file = {row["file"]: row for row in suite_manifest["tests"]}
    print(
        f"[triage]{lane_note} {len(targets)} targets, {len(done)} already done, "
        f"{len(todo)} to run, jobs={args.jobs} file_timeout={args.file_timeout}s "
        f"test_timeout={args.test_timeout}s",
        flush=True,
    )

    lock = __import__("threading").Lock()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(
                run_one,
                target,
                wt,
                args.py,
                args.file_timeout,
                args.test_timeout,
                args.markexpr,
                args.heavy_env,
                args.oom_fraction,
            ): target
            for target in todo
        }
        for fut in as_completed(futs):
            res = fut.result()
            file_record = suite_by_file[_node_file_part(res["file"])]
            res["tier"] = file_record["tier"]
            res["maturity"] = file_record["maturity"]
            res["owner"] = file_record["owner"]
            with lock:
                with out_path.open("a") as fh:
                    fh.write(json.dumps(res) + "\n")
                n_done += 1
                flag = "" if res["status"] == "PASS" else "  <-- " + res["status"]
                print(f"[{n_done}/{len(todo)}] {res['status']:11s} "
                      f"{res['peak_rss_mb']:8.1f}MB {res['elapsed_s']:7.1f}s "
                      f"{res['file']}{flag}", flush=True)

    print("[triage] DONE -> " + str(out_path), flush=True)


if __name__ == "__main__":
    main()
