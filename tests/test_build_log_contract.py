"""Contract tests for scripts/_build_log.sh — the quiet-build logging helper.

Born from the v0.15.131 release-gate post-mortem: three release-candidate
``build-test`` jobs failed undiagnosably because libint's codegen chatter
overflowed the GitLab runner's 4 MB trace cap ~35 minutes before the actual
error (which was then never collected — three times over). The helper routes
bulk build output to a log file, emits heartbeat lines, and on failure
replays the log tail with a resource snapshot, so the error always lands in
the visible trace.

These are pure shell-contract tests: no compiler, no network, no native
build. They drive the real helper through a bash child process.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = _REPO_ROOT / "scripts"
HELPER = SCRIPTS / "_build_log.sh"
NATIVE_STAMP = SCRIPTS / "_native_stamp.sh"


def _run(
    snippet: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run a snippet in bash with the helper sourced, set -euo pipefail on —
    the same shell regime build_libint.sh calls it under."""
    full_env = dict(os.environ)
    full_env.pop("VIBEQC_BUILD_VERBOSE", None)
    full_env.pop("VIBEQC_BUILD_LOG_HEARTBEAT_SECS", None)
    full_env.update(env or {})
    script = f"set -euo pipefail\n. {shlex.quote(str(HELPER))}\n{snippet}"
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=cwd,
    )


def test_success_path_redirects_bulk_output_and_restores_stdout(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo BULK-CHATTER\n"
        "vibeqc_build_log_end\n"
        "echo AFTER-END\n"
    )
    assert res.returncode == 0, res.stderr
    assert "BULK-CHATTER" not in res.stdout, res.stdout
    assert "BULK-CHATTER" in log.read_text(encoding="utf-8")
    assert "AFTER-END" in res.stdout, res.stdout
    assert re.search(r"demo build finished in \d+ min", res.stdout), res.stdout


def test_failure_replays_log_tail_and_preserves_exit_code(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo THE-ACTUAL-ERROR-LINE\n"
        "exit 7\n"
    )
    assert res.returncode == 7, (res.returncode, res.stdout, res.stderr)
    assert "demo build failed (exit 7)" in res.stdout, res.stdout
    assert "THE-ACTUAL-ERROR-LINE" in res.stdout, res.stdout
    assert "last 200 lines" in res.stdout, res.stdout
    # The post-mortem numbers the v0.15.131 incident lacked:
    assert "disk_avail=" in res.stdout, res.stdout


def test_failed_command_under_set_e_hits_the_trap(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo before-death\n"
        "false\n"
        "vibeqc_build_log_end\n"
    )
    assert res.returncode == 1, (res.returncode, res.stdout, res.stderr)
    assert "demo build failed (exit 1)" in res.stdout, res.stdout
    assert "before-death" in res.stdout, res.stdout


def test_verbose_opt_out_streams_and_writes_no_log(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo BULK-CHATTER\n"
        "vibeqc_build_log_end\n",
        env={"VIBEQC_BUILD_VERBOSE": "1"},
    )
    assert res.returncode == 0, res.stderr
    assert "BULK-CHATTER" in res.stdout, res.stdout
    assert not log.exists()


def test_heartbeat_reports_elapsed_and_last_ninja_step(tmp_path: Path):
    log = tmp_path / "build.log"
    res = _run(
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "echo '[12/100] Building CXX object foo.o'\n"
        "sleep 2.5\n"
        "vibeqc_build_log_end\n",
        env={"VIBEQC_BUILD_LOG_HEARTBEAT_SECS": "1"},
    )
    assert res.returncode == 0, res.stderr
    heartbeat = re.search(r"\[demo build \+\d+min\].*", res.stdout)
    assert heartbeat, res.stdout
    assert "[12/100]" in heartbeat.group(0), res.stdout
    assert "disk_avail=" in heartbeat.group(0), res.stdout


def test_heartbeat_orphans_pin_no_inherited_descriptors(tmp_path: Path):
    """An orphaned heartbeat sleep must not inherit the caller's long-lived
    descriptors. The concrete incident: build_libint.sh holds the build
    lock's FIFO write end at a high fd; an orphan inheriting it kept the
    lock-holder process — and with it the caller's stdout pipe — alive for
    a full heartbeat interval after the script had exited. Simulated here
    with a FIFO write end at fd 9 and a `cat` whose stdout is our pipe and
    which exits only when every write end of the FIFO is closed."""
    log = tmp_path / "build.log"
    fifo = tmp_path / "control"
    os.mkfifo(fifo)
    t0 = time.monotonic()
    res = _run(
        f"exec 9<> {shlex.quote(str(fifo))}\n"
        # 9>&-: the holder must not inherit the write end it is waiting on
        # (the real lock holder doesn't), or the test deadlocks on itself.
        f"cat {shlex.quote(str(fifo))} 9>&- &\n"
        f"vibeqc_build_log_begin {shlex.quote(str(log))} demo\n"
        "sleep 0.3\n"
        "vibeqc_build_log_end\n",
        env={"VIBEQC_BUILD_LOG_HEARTBEAT_SECS": "30"},
    )
    elapsed = time.monotonic() - t0
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert elapsed < 10, f"caller stayed pinned for {elapsed:.1f}s"


def test_quiet_build_plumbing_is_lifecycle_only():
    """The logging must not rotate libint's recipe hash: a rotation would
    rebuild libint (~hours at the 5_4_4 default) on every dev box and fleet
    host for a change that cannot alter the built artifact. The plumbing
    therefore lives inside vibeqc-recipe-hash lifecycle-only markers, and
    the hashed recipe content must keep the real build command while never
    seeing the logging calls."""
    res = subprocess.run(
        [
            "bash",
            "-c",
            "set -uo pipefail\n"
            f". {shlex.quote(str(SCRIPTS / '_libint_max_am.sh'))}\n"
            f". {shlex.quote(str(NATIVE_STAMP))}\n"
            f"_vibeqc_recipe_content {shlex.quote(str(SCRIPTS / 'build_libint.sh'))}\n",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    stripped = res.stdout
    assert "cmake --build" in stripped  # the real recipe stays hashed
    assert "vibeqc_build_log" not in stripped  # the logging never is
    assert "_build_log.sh" not in stripped


@pytest.mark.parametrize("fn", ["vibeqc_build_log_begin", "vibeqc_build_log_end"])
def test_helper_defines_its_entry_points(fn: str):
    text = HELPER.read_text(encoding="utf-8")
    assert f"{fn}()" in text


def _resume_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "native_resume", SCRIPTS / "_native_build_resume.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def resume_tree(tmp_path, monkeypatch):
    """Tiny tracked source and explicit fake compiler; no native science is built."""
    import shutil

    root = tmp_path / "product"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("build_libint.sh", "_native_stamp.sh", "_libint_max_am.sh"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    source = root / "third_party/libint/src"
    source.mkdir(parents=True)
    (source / "kernel.cc").write_text("int kernel() { return 1; }\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "kernel.cc"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    compiler = tmp_path / "compiler"
    compiler.write_text("#!/bin/sh\necho 12.2.0\n")
    compiler.chmod(0o755)
    monkeypatch.setenv("CXX", str(compiler))
    return root, source, compiler


def test_authenticated_resume_preserves_completed_objects(resume_tree, capsys):
    root, _, compiler = resume_tree
    module = _resume_module()
    module.prepare(root)
    assert capsys.readouterr().out == "fresh\n"
    _, _, build, _contract = module.locations(root)
    build.mkdir()
    (build / "CMakeCache.txt").write_text(f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n")
    (build / "build.ninja").write_text("# fake build graph\n")
    sentinel = build / "completed.o"
    sentinel.write_bytes(b"completed object preserved")
    module.configured(root)
    before = sentinel.stat().st_mtime_ns
    module.prepare(root)
    assert capsys.readouterr().out == "resume\n"
    assert sentinel.stat().st_mtime_ns == before
    assert sentinel.read_bytes() == b"completed object preserved"


@pytest.mark.parametrize(
    "change", ["source", "compiler", "configuration", "recipe", "environment"]
)
def test_resume_refuses_changed_inputs_without_deleting_objects(
    resume_tree, monkeypatch, change
):
    root, source, compiler = resume_tree
    module = _resume_module()
    module.prepare(root)
    _, _, build, _ = module.locations(root)
    build.mkdir()
    (build / "CMakeCache.txt").write_text(f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n")
    (build / "build.ninja").write_text("# graph\n")
    (build / "completed.o").write_bytes(b"keep")
    module.configured(root)
    if change == "source":
        (source / "kernel.cc").write_text("changed input")
    elif change == "compiler":
        compiler.write_text("different compiler")
    elif change == "configuration":
        with (build / "CMakeCache.txt").open("a") as out:
            out.write("CMAKE_CXX_FLAGS:STRING=-ffast-math\n")
    elif change == "recipe":
        with (root / "scripts/build_libint.sh").open("a") as out:
            out.write("echo changed-recipe\n")
    else:
        monkeypatch.setenv("CXXFLAGS", "-ffast-math")
    with pytest.raises(ValueError, match="changed"):
        module.prepare(root)
    assert (build / "completed.o").read_bytes() == b"keep"


def test_unrecorded_tree_requires_explicit_legacy_proof(resume_tree):
    root, _, compiler = resume_tree
    module = _resume_module()
    _, _, build, _ = module.locations(root)
    build.mkdir()
    (build / "CMakeCache.txt").write_text(f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n")
    with pytest.raises(ValueError, match="unrecorded"):
        module.prepare(root)
    assert (build / "CMakeCache.txt").exists()


def test_failed_resume_never_finalizes_native_markers(resume_tree, monkeypatch):
    root, _, _ = resume_tree
    module = _resume_module()
    proof = {"inputs": {"compiler": "fixture"}, "configuration": {}}
    monkeypatch.setattr(module, "verify_legacy", lambda *args: proof)
    monkeypatch.setattr(module, "held_build_lock", lambda *args: ())

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.recover(root, "a" * 40, True)
    dep, _, _, _ = module.locations(root)
    assert not (dep / "install/.vibeqc-max-am").exists()
    assert not (dep / "resume-result.json").exists()


def test_resume_lifecycle_preserves_published_libint_recipe():
    result = subprocess.run(
        [
            "bash",
            "-c",
            ". scripts/_native_stamp.sh; _vibeqc_recipe_hash scripts/build_libint.sh",
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == _resume_module().LEGACY_RECIPE


def test_nested_superbuild_cache_appears_after_first_configure(resume_tree, capsys):
    root, _, compiler = resume_tree
    module = _resume_module()
    module.prepare(root)
    _, _, build, _ = module.locations(root)
    build.mkdir()
    content = f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\nCMAKE_CXX_FLAGS:STRING=\n"
    (build / "CMakeCache.txt").write_text(content)
    (build / "build.ninja").write_text("# graph\n")
    module.configured(root)
    child = build / "library"
    child.mkdir()
    (child / "CMakeCache.txt").write_text(content)
    module.prepare(root)
    (child / "CMakeCache.txt").write_text(content + "INJECTED:BOOL=ON\n")
    with pytest.raises(ValueError, match="configuration changed"):
        module.prepare(root)


def test_legacy_bool_normalization_does_not_accept_changed_options():
    module = _resume_module()
    assert module.option_matches("0", "OFF")
    assert module.option_matches("ON", "ON")
    assert not module.option_matches("1", "OFF")
    assert not module.option_matches(None, "OFF")
    assert not module.option_matches("6;6;5", "6;6;6")


def test_legacy_recovery_refuses_wrong_product_identity(resume_tree, monkeypatch):
    root, _, _ = resume_tree
    module = _resume_module()
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: "wrong-sha")
    with pytest.raises(ValueError, match="immutable SHA"):
        module.verify_legacy(root, "a" * 40)


def test_legacy_recovery_refuses_unrecorded_compiler_environment(
    resume_tree, monkeypatch
):
    root, _, _ = resume_tree
    module = _resume_module()
    monkeypatch.setenv("CPATH", "/different/includes")
    with pytest.raises(ValueError, match="CPATH"):
        module.verify_legacy(root, "a" * 40)
