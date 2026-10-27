#!/usr/bin/env python3
"""Exercise an installed app with an empty home and no developer PATH."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def snapshot(app: Path) -> dict[str, str]:
    return {
        str(p.relative_to(app)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        for p in app.rglob("*")
        if p.is_file() and not p.is_symlink()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="queue-installed-") as temp:
        root = Path(temp)
        app = root / "Installed Apps With Spaces" / args.app.name
        shutil.copytree(args.app, app, symlinks=True)
        home = root / "empty-home"
        home.mkdir()
        env = {key: value for key, value in os.environ.items() if not key.startswith("VQ_")}
        env.update(
            HOME=str(home),
            PATH="/usr/bin:/bin:/usr/sbin:/sbin",
            XDG_CONFIG_HOME=str(home / "config"),
            XDG_DATA_HOME=str(home / "data"),
            XDG_CACHE_HOME=str(home / "cache"),
            VQ_STATE_DIR=str(home / "queue"),
            VQ_CONFIG_DIR=str(home / "config/vq"),
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
            VIRTUAL_ENV="/nonexistent",
            LD_LIBRARY_PATH="/nonexistent",
        )
        command = app / "bin/vq"
        before = snapshot(app)
        result = subprocess.run(
            [command, "--help"], cwd=home, env=env, check=True, text=True, capture_output=True
        )
        assert "web" in result.stdout and "submit" in result.stdout
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (root / "server.log").open("w+") as log:
            process = subprocess.Popen(
                [command, "web", "run", "--host", "127.0.0.1", "--port", str(port), "--no-fleet"],
                cwd=home,
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as res:
                            page = res.read().decode()
                            assert res.status == 200 and "<html" in page.lower()
                        break
                    except OSError:
                        if process.poll() is not None or time.monotonic() > deadline:
                            log.seek(0)
                            raise RuntimeError(log.read()) from None
                        time.sleep(0.2)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json") as res:
                    assert b'"openapi"' in res.read()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        assert snapshot(app) == before, "Running commands modified application resources"
        print(
            "PASS: relocated CLI and web console; isolated home, poisoned Python environment; "
            "application unchanged; server stopped"
        )


if __name__ == "__main__":
    main()
