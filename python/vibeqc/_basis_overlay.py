"""Private filesystem receipt for basis overlay freshness checks.

Setup pays the full content read. A receipt binds each SHA256 to the device,
inode, mode, size, modification time and POSIX status-change time of the file
actually read. Device numbers can differ between shared-filesystem clients;
that difference alone requires re-reading the content without changing the
receipt. Ordinary writes and replacements invalidate the remaining binding even
when callers restore the size and mtime. This is a filesystem cache, not a
security boundary against an actor who can forge the receipt or filesystem.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import NoReturn

_RECEIPT = ".basis-overlay-content-v1.json"
_MAX_RECEIPT_BYTES = 512 * 1024
_PARALLEL_VERIFY_BYTES = 4 * 1024 * 1024


class BasisOverlayIntegrityError(RuntimeError):
    """The selected generated basis catalog cannot be verified."""

    exit_status = 78


def _refuse(overlay: Path, detail: str) -> None:
    raise BasisOverlayIntegrityError(
        f"Basis overlay integrity refusal: {detail}\n"
        f"Overlay: {overlay}\n"
        "Regenerate with: bash scripts/setup_basis_library.sh (it needs "
        "Python 3.11+: set PYTHON_BIN if the project's .venv is absent). "
        "No integral calculation was started."
    )


def module_startup() -> bool:
    """True only while Python is locating our explicit ``-m`` target.

    During that phase argv[0] is the interpreter's ``-m`` sentinel; it
    becomes the target filename before user code runs. orig_argv retains
    the actual interpreter invocation, unlike arguments passed to a script.
    """
    import sys
    if sys.argv[:1] != ["-m"]:
        return False
    original = getattr(sys, "orig_argv", ())
    index = 1
    targets = ("vibeqc", "vibeqc.__main__", "vibeqc._cli", "vibeqc.cli")
    while index < len(original):
        option = original[index]
        if option == "--check-hash-based-pycs":
            index += 2
            continue
        if option == "--" or not option.startswith("-") or option == "-":
            return False
        if not option.startswith("--"):
            for offset, flag in enumerate(option[1:], 2):
                if flag == "c":
                    return False
                if flag in "mWX":
                    argument = option[offset:]
                    if not argument:
                        index += 1
                        argument = original[index] if index < len(original) else ""
                    if flag == "m":
                        return argument in targets
                    break  # -W / -X consume their entire option argument
        index += 1
    return False


def exit_at_module_startup(error: BasisOverlayIntegrityError) -> NoReturn:
    """Report a refusal raised while ``python -m vibeqc`` starts, then exit.

    No output channel exists yet during that import, so the message goes to
    stderr and the process exits with ``error.exit_status``. Call it only when
    :func:`module_startup` is true and re-raise otherwise, so a library import
    still sees the exception.
    """
    import sys
    print(str(error), file=sys.stderr)
    raise SystemExit(error.exit_status) from None


def _signature(stat: os.stat_result) -> list[int]:
    return [stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns]


def _load_records(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            encoded = stream.read(_MAX_RECEIPT_BYTES + 1)
        if len(encoded) > _MAX_RECEIPT_BYTES:
            return {}
        envelope = json.loads(encoded)
        payload = envelope["payload"]
        if hashlib.sha256(payload.encode()).hexdigest() != envelope["sha256"]:
            return {}
        decoded = json.loads(payload)
        if decoded["version"] != 1 or not isinstance(decoded["records"], dict):
            return {}
        return decoded["records"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def _save_records(path: Path, records: dict) -> None:
    temporary = None
    try:
        payload = json.dumps({"version": 1, "records": records}, separators=(",", ":"))
        envelope = json.dumps({"payload": payload,
                               "sha256": hashlib.sha256(payload.encode()).hexdigest()})
        if len(envelope.encode()) > _MAX_RECEIPT_BYTES:
            raise ValueError("basis receipt exceeds size limit")
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=".basis-overlay-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(envelope)
            os.fchmod(stream.fileno(), 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _content_record(path: str, previous: object) -> list:
    observed = _signature(os.stat(path))
    if not stat.S_ISREG(observed[2]):
        raise OSError("basis path is not a regular file")
    # Windows ctime can mean birth time. Coarse whole-second stamps cannot
    # safely distinguish nearby edits, so they do not enable cached hits.
    cacheable = os.name == "posix" and observed[-1] % 1_000_000_000 != 0
    if (cacheable and isinstance(previous, list) and len(previous) == 2
            and previous[0] == observed and isinstance(previous[1], str)
            and len(previous[1]) == 64
            and not previous[1].strip("0123456789abcdef")):
        return previous
    with open(path, "rb") as stream:
        before = _signature(os.fstat(stream.fileno()))
        value = hashlib.file_digest(stream, "sha256").hexdigest()
        after = _signature(os.fstat(stream.fileno()))
    if before != after or after != _signature(os.stat(path)):
        raise OSError("basis file changed during content verification")
    return [after, value]


def prepare_overlay_content(overlay: Path, bundled: Path, names: list[str]) -> None:
    """Setup authenticates actual bytes; ordinary imports never migrate data."""
    records = {}
    for name in names:
        records[name] = [_content_record(os.fspath(root / "basis" / name), None)
                         for root in (overlay, bundled)]
    _save_records(overlay / _RECEIPT, records)


def _verify_content_batch(overlay: Path, entries: list[tuple]) -> None:
    for path, name, observed, digest in entries:
        try:
            verified = _content_record(path, None)
        except OSError as exc:
            _refuse(overlay, f"{name}: cannot verify content ({exc})")
        if verified != [observed, digest]:
            _refuse(overlay, f"{name}: changed or unverifiable content record")


def _verify_content_parallel(overlay: Path, pending: list[tuple]) -> None:
    from threading import Thread

    errors: list[BaseException | None] = [None, None]

    def verify(side: int) -> None:
        try:
            _verify_content_batch(overlay, pending[side::2])
        except BaseException as exc:
            # A worker failure must never disappear as an uncaught thread
            # exception while the caller accepts an unverified catalog.
            errors[side] = exc

    started = []
    try:
        for side in range(2):
            worker = Thread(target=verify, args=(side,))
            worker.start()
            started.append(worker)
    except RuntimeError as exc:
        _refuse(overlay, f"cannot start content verification worker ({exc})")
    finally:
        for worker in started:
            worker.join()
    for error in errors:
        if error is not None:
            raise error


def compare_overlay_content(overlay: Path, bundled: Path, names: list[str]) -> list[str]:
    """Require a complete setup receipt and compare every prepared digest.

    Unknown freshness refuses selection, including missing/corrupt receipts,
    copied installations, unreadable files and changed file identities. A
    client-local device-number difference alone permits read-only content
    verification; every other identity field must still match the receipt.
    """
    old = _load_records(overlay / _RECEIPT)
    if not old:
        _refuse(overlay, "missing, corrupt or unreadable content receipt")
    differing = []
    pending = []
    roots = [os.fspath(root / "basis") + os.sep for root in (overlay, bundled)]
    for name in names:
        pair = old.get(name)
        if not isinstance(pair, list) or len(pair) != 2:
            _refuse(overlay, f"{name}: missing content record")
        for root, record in zip(roots, pair):
            try:
                observed = _signature(os.stat(root + name))
            except OSError as exc:
                _refuse(overlay, f"{name}: cannot read file identity ({exc.strerror})")
            if (os.name != "posix" or observed[-1] % 1_000_000_000 == 0
                    or not stat.S_ISREG(observed[2])
                    or not isinstance(record, list) or len(record) != 2
                    or not isinstance(record[0], list) or len(record[0]) != 6
                    or any(type(value) is not int for value in record[0])
                    or record[0][1:] != observed[1:] or not isinstance(record[1], str)
                    or len(record[1]) != 64
                    or record[1].strip("0123456789abcdef")):
                _refuse(overlay, f"{name}: changed or unverifiable content record")
            if record[0][0] != observed[0]:
                # st_dev is local to a filesystem client, so it cannot identify
                # a shared mount across clients. Prove the bytes rather than
                # trusting the remaining metadata or rewriting the receipt.
                pending.append((root + name, name, observed, record[1]))
        if pair[0][1] != pair[1][1]:
            differing.append(name)
    if len(pending) > 1 and sum(entry[2][3] for entry in pending) >= _PARALLEL_VERIFY_BYTES:
        # SHA256 releases the GIL. Two bounded batches overlap independent
        # content reads without a task/thread per file or persistent cache.
        # Small catalogs and same-client hits avoid worker startup entirely.
        _verify_content_parallel(overlay, pending)
    else:
        _verify_content_batch(overlay, pending)
    return differing


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record basis overlay content after setup")
    parser.add_argument("overlay", type=Path)
    parser.add_argument("bundled", type=Path)
    args = parser.parse_args()
    catalog = sorted(path.name for path in (args.bundled / "basis").iterdir()
                     if path.suffix in (".g94", ".ecp"))
    prepare_overlay_content(args.overlay, args.bundled, catalog)
    changed = compare_overlay_content(args.overlay, args.bundled, catalog)
    print(f"Basis content receipt: {len(catalog)} committed files; "
          f"{len(changed)} overlay files differ")
