"""Generated overlays must prove content before native basis use."""
from __future__ import annotations

import os
import re
import shutil

import pytest

import vibeqc
from vibeqc import _basis_overlay as guard

_CANARIES = ("def2-svp.g94", "6-311+g3df2p.g94", "pob-tzvp-rev2.g94")


def _pair(tmp_path):
    overlay, bundled = tmp_path / "overlay", tmp_path / "bundled"
    for root in (overlay, bundled):
        (root / "basis").mkdir(parents=True)
        for name in (*_CANARIES, "ordinary.g94", "unfamiliar.g94", "a.ecp", "z.ecp"):
            (root / "basis" / name).write_bytes(b"111")
    names = sorted(p.name for p in (bundled / "basis").iterdir())
    guard.prepare_overlay_content(overlay, bundled, names)
    return overlay, bundled, names


def test_prepared_overlay_and_legitimate_extras(tmp_path):
    overlay, bundled, _ = _pair(tmp_path)
    (overlay / "basis" / "extra.g94").write_bytes(b"extra")
    vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)
    assert (overlay / guard._RECEIPT).stat().st_mode & 0o777 == 0o644


def _other_filesystem_client(monkeypatch):
    """Observe real files through a client with a different device number."""
    signature = guard._signature

    def local_signature(info):
        observed = signature(info)
        observed[0] += 1
        return observed

    monkeypatch.setattr(guard, "_signature", local_signature)


def test_other_client_verifies_actual_bytes_without_resealing(tmp_path, monkeypatch):
    overlay, bundled, names = _pair(tmp_path)
    receipt = overlay / guard._RECEIPT
    before = receipt.read_bytes()
    _other_filesystem_client(monkeypatch)
    checked = []
    digest = guard.hashlib.file_digest

    def measure(stream, algorithm):
        checked.append(stream.name)
        return digest(stream, algorithm)

    monkeypatch.setattr(guard.hashlib, "file_digest", measure)
    monkeypatch.setattr(guard, "_save_records", lambda *args: pytest.fail("import resealed receipt"))
    overlay.chmod(0o555)
    try:
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)
    finally:
        overlay.chmod(0o755)
    assert set(checked) == {str(root / "basis" / name) for root in (overlay, bundled) for name in names}
    assert len(checked) == 2 * len(names)
    assert receipt.read_bytes() == before


def test_same_client_does_not_rehash_catalog(tmp_path, monkeypatch):
    overlay, bundled, _ = _pair(tmp_path)
    monkeypatch.setattr(guard.hashlib, "file_digest", lambda *args: pytest.fail("unchanged catalog rehashed"))
    vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)


@pytest.mark.parametrize("side", [0, 1])
def test_other_client_rejects_changed_bytes_even_with_matching_metadata(tmp_path, monkeypatch, side):
    overlay, bundled, names = _pair(tmp_path)
    target = (overlay, bundled)[side] / "basis" / "ordinary.g94"
    recorded = guard._signature(target.stat())
    target.write_bytes(b"222")
    signature = guard._signature

    def colliding_metadata(info):
        actual = signature(info)
        if actual[1] == recorded[1]:
            actual = recorded[:]
        actual[0] += 1
        return actual

    monkeypatch.setattr(guard, "_signature", colliding_metadata)
    with pytest.raises(guard.BasisOverlayIntegrityError, match="ordinary.g94"):
        guard.compare_overlay_content(overlay, bundled, names)


@pytest.mark.parametrize("race", ["write", "replace", "before_open"])
def test_other_client_refuses_file_changed_during_verification(tmp_path, monkeypatch, race):
    overlay, bundled, names = _pair(tmp_path)
    target = overlay / "basis" / names[0]
    _other_filesystem_client(monkeypatch)
    if race == "before_open":
        content_record = guard._content_record

        def changed_before_open(path, previous):
            target.write_bytes(b"222")
            target.write_bytes(b"111")
            return content_record(path, previous)

        monkeypatch.setattr(guard, "_content_record", changed_before_open)
    else:
        digest = guard.hashlib.file_digest

        def changed_during_read(stream, algorithm):
            value = digest(stream, algorithm)
            if race == "replace":
                replacement = overlay / "replacement"
                replacement.write_bytes(b"111")
                replacement.replace(target)
            else:
                target.write_bytes(b"222")
            return value

        monkeypatch.setattr(guard.hashlib, "file_digest", changed_during_read)
    with pytest.raises(guard.BasisOverlayIntegrityError, match=re.escape(names[0])):
        guard.compare_overlay_content(overlay, bundled, names)


@pytest.mark.parametrize("record", [None, ["bad", "0" * 64], [[1], "0" * 64],
                                  [["1"] * 6, "0" * 64], [[1] * 6, "not a digest"]])
def test_malformed_record_refuses(tmp_path, record):
    overlay, bundled, names = _pair(tmp_path)
    records = guard._load_records(overlay / guard._RECEIPT)
    records[names[0]][0] = record
    guard._save_records(overlay / guard._RECEIPT, records)
    with pytest.raises(guard.BasisOverlayIntegrityError, match=re.escape(names[0])):
        guard.compare_overlay_content(overlay, bundled, names)


@pytest.mark.parametrize("name", [*_CANARIES, "ordinary.g94", "unfamiliar.g94", "a.ecp", "z.ecp"])
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("other_client", [False, True])
def test_every_file_kind_same_size_restored_mtime_refuses(tmp_path, monkeypatch, name, side, other_client):
    overlay, bundled, _ = _pair(tmp_path)
    if other_client:
        _other_filesystem_client(monkeypatch)
    target = (overlay, bundled)[side] / "basis" / name
    before = target.stat()
    target.write_bytes(b"222")
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_size == before.st_size
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(guard.BasisOverlayIntegrityError, match=re.escape(name)):
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)


@pytest.mark.parametrize("state", ["missing", "corrupt", "oversized", "unreadable"])
def test_invalid_receipt_refuses_without_content_read_or_write(tmp_path, monkeypatch, state):
    overlay, bundled, _ = _pair(tmp_path)
    receipt = overlay / guard._RECEIPT
    if state == "missing":
        receipt.unlink()
    elif state == "corrupt":
        receipt.write_text("not json")
    elif state == "oversized":
        receipt.write_bytes(b"x" * (guard._MAX_RECEIPT_BYTES + 1))
    else:
        if not hasattr(os, "geteuid") or os.geteuid() == 0:
            pytest.skip("non-root POSIX permissions required")
        receipt.chmod(0)
    monkeypatch.setattr(guard, "_content_record", lambda *args: pytest.fail("import read basis content"))
    monkeypatch.setattr(guard, "_save_records", lambda *args: pytest.fail("import wrote receipt"))
    overlay.chmod(0o555)
    try:
        with pytest.raises(guard.BasisOverlayIntegrityError, match="receipt"):
            vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)
    finally:
        overlay.chmod(0o755)
        if receipt.exists():
            receipt.chmod(0o644)


def test_missing_file_names_refusal(tmp_path):
    overlay, bundled, _ = _pair(tmp_path)
    (overlay / "basis" / "z.ecp").unlink()
    with pytest.raises(guard.BasisOverlayIntegrityError, match="z.ecp"):
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)


def test_prepared_different_bytes_refuse(tmp_path):
    overlay, bundled, names = _pair(tmp_path)
    (bundled / "basis" / "ordinary.g94").write_bytes(b"222")
    guard.prepare_overlay_content(overlay, bundled, names)
    with pytest.raises(guard.BasisOverlayIntegrityError, match="different content: ordinary.g94"):
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)


@pytest.mark.parametrize("other_client", [False, True])
def test_copied_receipt_and_atomic_replacement_refuse(tmp_path, monkeypatch, other_client):
    overlay, bundled, _ = _pair(tmp_path)
    if other_client:
        _other_filesystem_client(monkeypatch)
    copy = tmp_path / "copied"
    shutil.copytree(overlay, copy)
    with pytest.raises(guard.BasisOverlayIntegrityError):
        vibeqc._warn_if_basis_overlay_is_stale(copy, bundled)
    target = overlay / "basis" / "ordinary.g94"
    previous = target.stat()
    replacement = overlay / "replacement"
    replacement.write_bytes(b"222")
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    replacement.replace(target)
    with pytest.raises(guard.BasisOverlayIntegrityError, match="ordinary.g94"):
        vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)


def test_unreadable_file_and_absent_bundled_catalog_refuse(tmp_path):
    overlay, bundled, _ = _pair(tmp_path)
    target = overlay / "basis" / "a.ecp"
    target.chmod(0)
    try:
        with pytest.raises(guard.BasisOverlayIntegrityError, match="a.ecp"):
            vibeqc._warn_if_basis_overlay_is_stale(overlay, bundled)
    finally:
        target.chmod(0o644)
    with pytest.raises(guard.BasisOverlayIntegrityError, match="cannot inspect"):
        vibeqc._warn_if_basis_overlay_is_stale(overlay, tmp_path / "absent")


@pytest.mark.parametrize("valid", [True, False])
def test_explicit_user_library_is_not_overridden_or_checked(tmp_path, monkeypatch, valid):
    user = tmp_path / "user"
    if valid:
        (user / "basis").mkdir(parents=True)
        for name in _CANARIES:
            (user / "basis" / name).write_bytes(b"111")
    monkeypatch.setenv("LIBINT_DATA_PATH", str(user))
    monkeypatch.setattr(vibeqc, "_resolve_basis_library", lambda: pytest.fail("user path replaced"))
    vibeqc._install_basis_library()
    assert os.environ["LIBINT_DATA_PATH"] == str(user)


def test_this_clone_overlay_carries_the_pp_family():
    """The concrete regression: whatever the resolution picks must have the
    PP orbital sets, or heavy-element ECP work fails with a message that
    blames the basis name rather than the stale overlay."""
    root = vibeqc._resolve_basis_library()
    if root is None:
        pytest.skip("no basis library resolved in this environment")
    missing = [
        f"{aug}cc-p{core}{zeta}-pp{ext}"
        for zeta in ("dz", "tz", "qz", "5z")
        for core in ("v", "wcv")
        for aug in ("", "aug-")
        for ext in (".g94", ".ecp")
        if not (root / "basis" / f"{aug}cc-p{core}{zeta}-pp{ext}").is_file()
    ]
    assert not missing, (
        f"{root / 'basis'} is missing {len(missing)} PP file(s), e.g. "
        f"{missing[:3]}: this library predates the 2026-09-08 import. "
        "Run: bash scripts/setup_basis_library.sh"
    )


def test_this_clone_resolves_a_library_that_actually_carries_silver():
    """The regression itself: whatever the resolution picks must have the
    heavy def2 elements #88 added, or ECP work silently resolves pre-#88
    data. Reads the resolved library rather than the committed one."""
    root = vibeqc._resolve_basis_library()
    if root is None:
        pytest.skip("no basis library resolved in this environment")
    text = (root / "basis" / "def2-svp.g94").read_text()
    assert "\nAg     0\n" in text, (
        f"{root / 'basis' / 'def2-svp.g94'} has no silver block: this "
        "library predates #88. Run: bash scripts/setup_basis_library.sh"
    )


def test_cli_startup_status_and_normal_library_exception(tmp_path):
    """Exercise real interpreter startup, never infer it from script arguments."""
    import inspect
    import subprocess
    import sys
    import vibeqc_bootstrap

    package = tmp_path / "vibeqc"
    package.mkdir()
    (package / "_basis_overlay.py").write_text(inspect.getsource(guard))
    (package / "__init__.py").write_text(
        "import sys\nfrom ._basis_overlay import BasisOverlayIntegrityError, module_startup\n"
        "try:\n raise BasisOverlayIntegrityError('ordinary.g94: refusal')\n"
        "except BasisOverlayIntegrityError as exc:\n"
        " if module_startup():\n  raise SystemExit(exc.exit_status)\n"
        " raise\n"
    )
    (package / "__main__.py").write_text("raise AssertionError('entry reached')\n")
    (tmp_path / "vibeqc_bootstrap.py").write_text(inspect.getsource(vibeqc_bootstrap))
    code = ("try:\n import vibeqc\nexcept RuntimeError as exc:\n"
            " assert type(exc).__name__ == 'BasisOverlayIntegrityError'\n"
            "else:\n raise AssertionError('no refusal')")
    (tmp_path / "other.py").write_text(code)
    other_package = tmp_path / "other_package"
    other_package.mkdir()
    (other_package / "__init__.py").write_text(code)
    (other_package / "__main__.py").write_text("pass\n")
    cases = [(["-m", "vibeqc"], 78), (["-B", "-X", "dev", "-m", "vibeqc"], 78),
             (["-c", "import vibeqc_bootstrap; raise SystemExit(vibeqc_bootstrap.main())"], 78),
             (["-c", code], 0), (["-c", code, "-m", "vibeqc"], 0), (["-m", "other"], 0),
             (["-m", "other_package", "-m", "vibeqc"], 0),
             (["-X", "-m", "-m", "vibeqc"], 78), (["-Bm", "vibeqc"], 78)]
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    for argv, status in cases:
        # This fixture uses only the standard library and its synthetic package.
        # Editable-install site hooks can otherwise intercept that package name
        # before PYTHONPATH and accidentally launch the real application.
        result = subprocess.run([sys.executable, "-S", *argv], cwd=tmp_path, env=env,
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == status, (argv, result.stderr)


@pytest.mark.parametrize("selection", ["project", "active", "old_project", "broken_project", "path", "explicit"])
def test_basis_setup_selects_supported_python_and_writes_real_receipt(tmp_path, selection):
    root, env, invoke, log = _basis_setup_fixture(tmp_path)
    if selection in {"project", "old_project", "broken_project", "explicit"}:
        mode = {"old_project": "old", "broken_project": "broken"}.get(selection, "good")
        invoke(root / ".venv/bin/python", "project", mode)
    if selection in {"project", "active", "old_project", "broken_project", "explicit"}:
        invoke(tmp_path / "active env/bin/python", "active")
        env["VIRTUAL_ENV"] = str(tmp_path / "active env")
    invoke(tmp_path / "bin/python3", "path", "good" if selection == "path" else "old")
    if selection == "explicit":
        override = tmp_path / "explicit python"
        invoke(override, "explicit")
        env["PYTHON_BIN"] = str(override)
    result = _run_basis_setup(root, env)
    assert result.returncode == 0, result.stderr
    chosen = "active" if selection in {"active", "old_project", "broken_project"} else selection
    calls = log.read_text().splitlines()
    assert calls[-2:] == [chosen + ":script", chosen + ":script"], calls
    target, bundled = root / "build/basis_library", root / "python/vibeqc/basis_library"
    for name in ("minimal.g94", "minimal.ecp"):
        assert (target / "basis" / name).read_bytes() == (bundled / "basis" / name).read_bytes()
    assert guard.compare_overlay_content(target, bundled, ["minimal.g94", "minimal.ecp"]) == []
    assert "Basis content receipt: 2 committed files; 0 overlay files differ" in result.stdout


@pytest.mark.parametrize("failure", ["old_path", "missing_path", "old_explicit", "missing_explicit"])
@pytest.mark.parametrize("existing", [False, True])
def test_basis_setup_refuses_unsupported_python_before_any_write(tmp_path, failure, existing):
    root, env, invoke, log = _basis_setup_fixture(tmp_path)
    if failure == "old_path":
        invoke(tmp_path / "bin/python3", "path", "old")
    if failure.endswith("explicit"):
        # A valid project interpreter must never override an explicit refusal.
        invoke(root / ".venv/bin/python", "project")
        override = tmp_path / "explicit python"
        if failure == "old_explicit":
            invoke(override, "explicit", "old")
        env["PYTHON_BIN"] = str(override)
    target = root / "build/basis_library"
    if existing:
        (target / "basis").mkdir(parents=True)
        (target / "basis/sentinel").write_bytes(b"keep existing overlay")
        (target / guard._RECEIPT).write_bytes(b"keep existing receipt")
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = _run_basis_setup(root, env)
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert result.returncode != 0
    assert "PYTHON_BIN=" in result.stderr and ">=3.11" in result.stderr
    assert env.get("PYTHON_BIN", "python3") in result.stderr
    assert after == before
    assert target.exists() == existing
    if log.exists():
        assert all(call.endswith(":probe") for call in log.read_text().splitlines())


def _run_basis_setup(root, env):
    import subprocess
    return subprocess.run(["/bin/bash", str(root / "scripts/setup_basis_library.sh")],
                          env=env, cwd=root, capture_output=True, text=True, timeout=15)


def _basis_setup_fixture(tmp_path):
    """Actual setup/split/receipt files; interpreter version probes are doubles."""
    import shlex
    import sys
    from pathlib import Path

    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "project with spaces"
    for name in ("scripts/setup_basis_library.sh", "scripts/basisset_dev/split_ecp_g94.py",
                 "python/vibeqc/_basis_overlay.py"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    standard = root / "third_party/libint/install/share/libint/fixture/basis"
    standard.mkdir(parents=True)
    orbital = "H 0\nS 1 1.00\n  2.0 1.0\n****\n"
    sidecar = "Na-ECP 2 10\nul potential\n1\n2 1.0 2.0\n"
    (standard / "minimal.g94").write_text(orbital.replace("2.0", "1.0"))
    for part in ("custom", "basis"):
        library = root / "python/vibeqc/basis_library" / part
        library.mkdir(parents=True)
        (library / "minimal.g94").write_text(orbital)
        (library / "minimal.ecp").write_text(sidecar)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    # Keep the missing-python case independent of the invoking host's PATH.
    for command in ("dirname", "find", "head", "mkdir", "rm", "cp", "ls", "wc", "tr"):
        (binaries / command).symlink_to(shutil.which(command))
    log = tmp_path / "interpreter-calls"
    env = dict(os.environ, PATH=str(binaries), PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="")
    env.pop("PYTHON_BIN", None)
    env.pop("VIRTUAL_ENV", None)

    def invoke(path, label, mode="good"):
        path.parent.mkdir(parents=True, exist_ok=True)
        body = '#!/bin/bash\nkind=script\n[ "$1" != "-c" ] || kind=probe\n'
        body += 'printf "%s:%s\\n" ' + shlex.quote(label) + ' "$kind" >> ' + shlex.quote(str(log)) + '\n'
        if mode == "old":
            body += 'if [ "$1" = "-c" ]; then echo 3.9.0; exit 1; fi\n'
        elif mode == "broken":
            body += 'exit 126\n'
        body += 'exec ' + shlex.quote(sys.executable) + ' "$@"\n'
        path.write_text(body)
        path.chmod(0o755)

    return root, env, invoke, log
