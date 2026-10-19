"""Copy QC's complete native library closure and replace build-machine paths."""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from runtime import MACHO, digest, run


def inspect(path: Path) -> tuple[list[str], list[str], str | None]:
    lines = run(["/usr/bin/otool", "-L", str(path)]).splitlines()[1:]
    dependencies = [line.strip().split(" (", 1)[0] for line in lines]
    ids = run(["/usr/bin/otool", "-D", str(path)]).splitlines()[1:]
    commands = run(["/usr/bin/otool", "-l", str(path)])
    rpaths = re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.*?) \(offset", commands)
    return dependencies, rpaths, ids[0] if ids else None


def build_fftw(cache: Path, source: Path) -> Path:
    """Build the small FFTW dependency with portable diagnostic source paths."""
    target = cache / "fftw-runtime"
    identity = {"sourceSha256": digest(source), "recipeSha256": digest(Path(__file__))}
    receipt = target / "receipt.json"
    library = target / "install/lib/libfftw3.3.dylib"
    if receipt.exists() and library.exists():
        previous = json.loads(receipt.read_text())
        if all(previous.get(k) == v for k, v in identity.items()) and previous.get(
            "librarySha256"
        ) == digest(library):
            return target / "install"
    with tempfile.TemporaryDirectory(prefix="fftw-package-", dir=cache) as temp:
        work = Path(temp)
        with tarfile.open(source) as archive:
            archive.extractall(work, filter="data")
        src = work / "fftw-3.3.10"
        run(
            [
                "cmake",
                "-S",
                str(src),
                "-B",
                str(work / "build"),
                "-GNinja",
                "-Wno-dev",
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_INSTALL_PREFIX={work / 'install'}",
                f"-DCMAKE_C_FLAGS=-ffile-prefix-map={src}=fftw-3.3.10",
                "-DBUILD_TESTS=OFF",
                "-DBUILD_SHARED_LIBS=ON",
                "-DENABLE_OPENMP=ON",
                "-DENABLE_THREADS=ON",
            ]
        )
        run(
            [
                "cmake",
                "--build",
                str(work / "build"),
                "--target",
                "install",
                "--parallel",
                "4",
            ]
        )
        identity["librarySha256"] = digest(work / "install/lib/libfftw3.3.dylib")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()
        (work / "install").rename(target / "install")
        receipt.write_text(json.dumps(identity, indent=2) + "\n")
    return target / "install"


def relocate(prefix: Path, native_prefix: Path, fftw_install: Path) -> None:
    destination = prefix / "lib/vibe-native"
    destination.mkdir()
    search = [
        path
        for dep in ("libint", "libxc", "spglib", "fftw", "libecpint")
        for path in (
            native_prefix / dep / "install/lib",
            native_prefix / dep / "install/lib64",
        )
    ]
    copied: dict[Path, Path] = {}
    inventory: list[dict[str, str]] = []

    def system(path: str) -> bool:
        return path.startswith(("/usr/lib/", "/System/Library/"))

    def visit(source: Path, target: Path) -> None:
        source = source.resolve()
        if source in copied:
            return
        copied[source] = target
        before = digest(source)
        dependencies, rpaths, identity = inspect(source)
        changes = []
        for dependency in dependencies:
            if dependency == identity or system(dependency):
                continue
            if Path(dependency).name.startswith("libfftw3."):
                candidates = [fftw_install / "lib/libfftw3.3.dylib"]
            elif dependency.startswith("@loader_path/"):
                candidates = [source.parent / dependency.removeprefix("@loader_path/")]
            elif dependency.startswith("@rpath/"):
                name = dependency.removeprefix("@rpath/")
                candidates = [
                    Path(p.replace("@loader_path", str(source.parent))) / name
                    for p in rpaths
                ] + [p / name for p in search]
            else:
                candidates = [Path(dependency)]
            dependency_source = next(
                (p.resolve() for p in candidates if p.is_file()), None
            )
            if dependency_source is None:
                raise ValueError(
                    f"Missing native dependency: {source.name}: {dependency}"
                )
            if not (
                dependency_source.is_relative_to(native_prefix.resolve())
                or dependency_source.is_relative_to(fftw_install.resolve())
                or dependency_source.name == "libomp.dylib"
            ):
                raise ValueError(
                    f"Unreviewed native dependency: {dependency_source.name}"
                )
            dependency_target = destination / dependency_source.name
            if dependency_source not in copied:
                if dependency_target.exists():
                    raise ValueError(
                        f"Native library basename collision: {dependency_target.name}"
                    )
                shutil.copyfile(dependency_source, dependency_target)
                dependency_target.chmod(0o755)
                visit(dependency_source, dependency_target)
            replacement = "@loader_path/" + os.path.relpath(
                dependency_target, target.parent
            )
            changes.extend(["-change", dependency, replacement])
        for path in rpaths:
            changes.extend(["-delete_rpath", path])
        if identity:
            changes.extend(["-id", "@rpath/" + target.name])
        if changes:
            run(["/usr/bin/install_name_tool", *changes, str(target)])
        if re.search(rb"/Users/[A-Za-z][A-Za-z0-9_.-]*/", target.read_bytes()):
            raise ValueError(
                f"{target.name} embeds a user-home build path; rebuild with portable source/data paths"
            )
        run(["/usr/bin/codesign", "--force", "--sign", "-", str(target)])
        inventory.append(
            {
                "file": str(target.relative_to(prefix)),
                "inputSha256": before,
                "bundledSha256": digest(target),
            }
        )

    cores = list(prefix.glob("lib/python*/site-packages/vibeqc/_vibeqc_core*.so"))
    if len(cores) != 1:
        raise ValueError("Expected exactly one installed QC native extension")
    visit(cores[0], cores[0])
    (prefix.parent / "native-libraries.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )


def sanitize_search_paths(prefix: Path) -> None:
    """Remove unused build-host search directories left in delocated wheels."""
    for path in prefix.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as stream:
            if stream.read(4) not in MACHO:
                continue
        _, rpaths, _ = inspect(path)
        changes = []
        for rpath in rpaths:
            if rpath.startswith("/") and not rpath.startswith(
                ("/usr/lib/", "/System/Library/")
            ):
                changes.extend(["-delete_rpath", rpath])
        if changes:
            run(["/usr/bin/install_name_tool", *changes, str(path)])
            run(["/usr/bin/codesign", "--force", "--sign", "-", str(path)])
    # check_native subsequently proves that all remaining dependencies resolve
    # inside the app; removing a path never silently admits a missing library.


def collect_native_sources(native_prefix: Path, notices: Path) -> None:
    """Preserve build configuration; upstream sources and patches ship separately."""
    for name in ("libint", "libxc", "spglib", "fftw", "libecpint"):
        install = native_prefix / name / "install"
        target = notices / "native-build-inputs" / name
        target.mkdir(parents=True)
        for marker in install.glob(".vibeqc-*"):
            shutil.copyfile(marker, target / marker.name)
    # The reusable dependency recipes and patches are in the application source
    # archive. Upstream source tarballs are mandatory hash-locked notice inputs.
