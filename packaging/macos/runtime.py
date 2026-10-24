"""Hash verification and native checks shared with vibe-view (MPL-2.0)."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MACHO = {
    b"\xcf\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
}
NOTICE = re.compile(r"license|licence|copyright|copying|notice|authors", re.IGNORECASE)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args: list[str], **kwargs: Any) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)
    return str(result.stdout)


def clean_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "DYLD_"))
        and key not in {"VIRTUAL_ENV", "__PYVENV_LAUNCHER__", "RDBASE"}
    }


def fetch(item: dict[str, Any], cache: Path) -> Path:
    name = str(item["filename"])
    if Path(name).name != name:
        raise ValueError(f"Invalid input filename: {name}")
    dest = cache / name
    if not dest.exists():
        if not item["url"].startswith("https://"):
            raise ValueError("Runtime inputs require HTTPS")
        partial = dest.with_suffix(dest.suffix + ".partial")
        try:
            with (
                urllib.request.urlopen(item["url"], timeout=180) as response,
                partial.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            if digest(partial) != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {name}")
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
    if digest(dest) != item["sha256"]:
        raise ValueError(f"Checksum mismatch in cache: {name}; remove it and retry")
    return dest


def collect_notices(archive: Path, destination: Path) -> None:
    """Copy license texts only, without following archive links."""
    count = 0
    with tarfile.open(archive) as source:
        for member in source:
            if not member.isfile() or not NOTICE.search(Path(member.name).name):
                continue
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            assert stream is not None
            with stream, target.open("wb") as output:
                shutil.copyfileobj(stream, output)
            count += 1
    if not count:
        raise ValueError(f"No license notices found in {archive.name}")


def check_native(root: Path, minimum_macos: str) -> int:
    """Reject wrong architectures and references to build-machine libraries."""
    binaries = []
    for file in root.rglob("*"):
        if file.is_symlink():
            if not file.resolve().is_relative_to(root.resolve()) or not file.exists():
                raise ValueError(f"Non-relocatable symlink: {file.relative_to(root)}")
            continue
        if not file.is_file():
            continue
        with file.open("rb") as stream:
            if stream.read(4) not in MACHO:
                continue
        binaries.append(file)

    def check(file: Path) -> None:
        commands = run(["/usr/bin/otool", "-arch", "arm64", "-l", str(file)])
        rpaths = re.findall(
            r"cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset", commands
        )
        for path in rpaths:
            if path.startswith("/") and not path.startswith(
                ("/usr/lib/", "/System/Library/")
            ):
                raise ValueError(f"External search path in {file.relative_to(root)}")
        minimum = re.search(r"\bminos ([\d.]+)", commands)
        if minimum and tuple(map(int, (minimum[1] + ".0.0").split(".")[:3])) > tuple(
            map(int, (minimum_macos + ".0.0").split(".")[:3])
        ):
            raise ValueError(
                f"{file.name} requires macOS {minimum[1]}, target is {minimum_macos}"
            )

        def expand(path: str) -> Path:
            return Path(
                path.replace("@loader_path", str(file.parent)).replace(
                    "@executable_path", str(root / "bin")
                )
            )

        archs = run(["/usr/bin/lipo", "-archs", str(file)]).split()
        if "arm64" not in archs:
            raise ValueError(f"Missing arm64 code: {file.relative_to(root)}")
        # LC_ID_DYLIB can be a bare name (not a load dependency).
        identity = run(
            ["/usr/bin/otool", "-arch", "arm64", "-D", str(file)]
        ).splitlines()[1:]
        for line in run(
            ["/usr/bin/otool", "-arch", "arm64", "-L", str(file)]
        ).splitlines()[1:]:
            dependency = line.strip().split(" (", 1)[0]
            if dependency in identity:
                continue
            if not dependency.startswith(
                (
                    "@loader_path/",
                    "@rpath/",
                    "@executable_path/",
                    "/usr/lib/",
                    "/System/Library/",
                )
            ):
                raise ValueError(
                    f"External library in {file.relative_to(root)}: {dependency}"
                )
            if dependency.startswith("@rpath/"):
                candidates = [
                    expand(p) / dependency.removeprefix("@rpath/") for p in rpaths
                ]
                candidates.append(root / "lib" / dependency.removeprefix("@rpath/"))
            elif dependency.startswith("@"):
                candidates = [expand(dependency)]
            else:
                continue  # Operating-system library, including the dyld shared cache.
            if not any(
                p.exists() and p.resolve().is_relative_to(root.resolve())
                for p in candidates
            ):
                raise ValueError(
                    f"Unresolved internal library in {file.name}: {dependency}"
                )

    if not binaries:
        raise ValueError("No native Python runtime found")
    # Independent native inspections dominate build time; bound their fan-out.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, binaries))
    return len(binaries)
