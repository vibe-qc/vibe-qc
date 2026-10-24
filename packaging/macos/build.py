#!/usr/bin/env python3
"""Build a self-contained Apple Silicon app, ZIP and DMG from locked inputs."""

from __future__ import annotations

import argparse
import json
import platform
import plistlib
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path

from runtime import check_native, clean_environment, collect_notices, digest, fetch, run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = HERE / "runtime.lock.json"


def verify(app: Path, lock: dict) -> None:
    resources = app / "Contents/Resources"
    manifest = json.loads((resources / "runtime.json").read_text())
    if manifest["lockSha256"] != digest(LOCK):
        raise ValueError("Runtime lock differs from this builder")
    code = (
        lock["probe"]
        + "\n"
        + """
import json, sys, pathlib, importlib.metadata as m
prefix = pathlib.Path(sys.prefix).resolve()
distributions = list(m.distributions())
assert all(pathlib.Path(d.locate_file('')).resolve().is_relative_to(prefix) for d in distributions)
print(json.dumps({'python': sys.version.split()[0],
                 'packages': {d.metadata['Name'].lower(): d.version for d in distributions}}))
"""
    )
    # A probe must never consult the packager's queue, credentials or caches.
    with tempfile.TemporaryDirectory(prefix="package-probe-") as home:
        env = {k: v for k, v in clean_environment().items() if not k.startswith("VQ_")}
        env.update(
            HOME=home,
            XDG_CONFIG_HOME=home + "/config",
            XDG_DATA_HOME=home + "/data",
            XDG_CACHE_HOME=home + "/cache",
            VQ_CONFIG_DIR=home + "/config/vq",
            VQ_STATE_DIR=home + "/queue",
            PATH="/usr/bin:/bin:/usr/sbin:/sbin",
        )
        for key in ("LIBINT_DATA_PATH", "VQ_DATA_DIR"):
            env.pop(key, None)
        actual = json.loads(
            run(
                [str(resources / "python/bin/python3"), "-I", "-B", "-c", code],
                cwd=home,
                env=env,
            )
        )
    expected = {p["name"].lower(): p["version"] for p in lock["wheels"]}
    expected[lock["distribution"]] = manifest["appVersion"]
    if actual != {"python": lock["pythonVersion"], "packages": expected}:
        raise ValueError(f"Unexpected runtime inventory: {actual}")
    count = check_native(resources / "python", lock["minimumMacOS"])
    print(
        f"Verified {app.name}: {len(expected)} distributions, {count} native files",
        flush=True,
    )


def build(args: argparse.Namespace, lock: dict) -> None:
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {
        i["filename"]: fetch(i, args.cache)
        for i in [
            lock["python"],
            lock["pythonNotices"],
            *lock["wheels"],
            *lock["buildWheels"],
            *lock["nativeNotices"],
        ]
    }
    revision = run(["git", "rev-parse", args.source_revision], cwd=ROOT).strip()
    project = tomllib.loads(
        run(["git", "show", f"{revision}:pyproject.toml"], cwd=ROOT)
    )["project"]
    version = project["version"]
    with tempfile.TemporaryDirectory(prefix="macos-package-", dir=args.output) as temp:
        work = Path(temp)
        app = work / f"{lock['appName']}.app"
        resources = app / "Contents/Resources"
        macos = app / "Contents/MacOS"
        resources.mkdir(parents=True)
        macos.mkdir()
        with tarfile.open(inputs[lock["python"]["filename"]]) as archive:
            archive.extractall(resources, filter="data")
        python = resources / "python/bin/python3"
        pip = [
            str(python),
            "-I",
            "-B",
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
        ]
        env = clean_environment()
        if args.app_wheel:
            wheel = args.app_wheel.resolve()
        else:
            run(
                [
                    *pip,
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--no-compile",
                    *[str(inputs[i["filename"]]) for i in lock["buildWheels"]],
                ],
                env=env,
            )
            # Export only wheel inputs, not a checkout, and never include another
            # session's uncommitted edits, user config or ignored build products.
            archive = work / "wheel-inputs.tar"
            run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    f"--output={archive}",
                    revision,
                    *lock["exportPaths"],
                ],
                cwd=ROOT,
            )
            source = work / "wheel-inputs"
            source.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(source, filter="data")
            wheels = work / "wheels"
            run(
                [
                    *pip,
                    "wheel",
                    "--no-index",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheels),
                    str(source),
                ],
                env=env,
            )
            (wheel,) = wheels.glob("*.whl")
        run(
            [
                *pip,
                "install",
                "--no-index",
                "--no-deps",
                "--no-compile",
                str(wheel),
                *[str(inputs[i["filename"]]) for i in lock["wheels"]],
            ],
            env=env,
        )
        run([*pip, "check"], env=env)
        extras = ",".join(lock["extras"])
        run(
            [
                *pip,
                "install",
                "--dry-run",
                "--no-index",
                f"{lock['distribution']}[{extras}]",
            ],
            env=env,
        )
        build_only = {i["name"] for i in lock["buildWheels"]} - {
            i["name"] for i in lock["wheels"]
        }
        run(
            [
                *pip,
                "uninstall",
                "-y",
                "pip",
                "setuptools",
                "wheel",
                *sorted(build_only),
            ],
            env=env,
        )
        prefix = resources / "python"
        for file in (prefix / "bin").iterdir():
            if not re.fullmatch(r"python(?:3(?:\.\d+)?)?", file.name):
                file.unlink()
        for name in ("include", "share", "lib/pkgconfig"):
            shutil.rmtree(prefix / name, ignore_errors=True)
        for path in prefix.rglob("__pycache__"):
            shutil.rmtree(path)
        for pattern in ("*.pyc", "*.a", "direct_url.json"):
            for path in prefix.rglob(pattern):
                path.unlink()
        lib = prefix / "lib" / f"python{lock['pythonMinor']}"
        shutil.rmtree(lib / "test", ignore_errors=True)
        for path in lib.glob("config-*"):
            shutil.rmtree(path)
        if lock.get("nativeLibraries"):
            from native import build_fftw, relocate, sanitize_search_paths

            fftw_install = build_fftw(args.cache, inputs["fftw-source.tar.gz"])
            relocate(prefix, args.native_prefix, fftw_install)
            sanitize_search_paths(prefix)
        notices = resources / "notices"
        notices.mkdir()
        if lock.get("nativeLibraries"):
            from native import collect_native_sources

            collect_native_sources(args.native_prefix, notices)
        run(
            [
                "/usr/bin/tar",
                "-xf",
                str(inputs[lock["pythonNotices"]["filename"]]),
                "-C",
                str(notices),
                "python/PYTHON.json",
                "python/licenses",
            ]
        )
        for item in lock["nativeNotices"]:
            source = inputs[item["filename"]]
            target = notices / item["component"]
            target.mkdir(exist_ok=True)
            if item["archive"]:
                collect_notices(source, target)
            else:
                shutil.copyfile(source, target / source.name)
            if item.get("shipSource"):
                shutil.copyfile(source, target / source.name)
        build_sources = notices / "package-builder"
        build_sources.mkdir()
        for path in HERE.iterdir():
            if path.is_file() and path.suffix in {".py", ".c", ".json", ".md"}:
                shutil.copyfile(path, build_sources / path.name)
        shutil.copyfile(LOCK, notices / "runtime-lock.json")
        shutil.copyfile(
            ROOT / "LICENSE", notices / f"{lock['distribution']}-LICENSE.txt"
        )
        shutil.copyfile(HERE / "README.md", resources / "README.md")
        # The source supplement is delivered with every download, including paid builds.
        source_archive = (
            notices / f"{lock['distribution']}-{revision[:12]}-source.tar.gz"
        )
        run(
            [
                "git",
                "archive",
                "--format=tar.gz",
                f"--output={source_archive}",
                revision,
            ],
            cwd=ROOT,
        )
        manifest = {
            "format": 1,
            "appVersion": version,
            "sourceRevision": revision,
            "pythonVersion": lock["pythonVersion"],
            "minimumMacOS": lock["minimumMacOS"],
            "lockSha256": digest(LOCK),
            "appWheelSha256": digest(wheel),
            "sourceSha256": digest(source_archive),
            "entrypoints": project["scripts"],
            "packages": {p["name"]: p["version"] for p in lock["wheels"]},
        }
        (resources / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (
            resources / "run_entrypoint.py"
        ).write_text("""import importlib.metadata as metadata
import json, pathlib, sys
manifest = json.loads(pathlib.Path(__file__).with_name("runtime.json").read_text())
name = sys.argv.pop(1)
if name not in manifest["entrypoints"]:
    raise SystemExit("Unknown bundled command: " + name)
entry = next(e for e in metadata.entry_points(group="console_scripts") if e.name == name)
sys.argv[0] = name
raise SystemExit(entry.load()())
""")
        (resources / "Start.command").write_text("""#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PATH="$HERE/../MacOS:$PATH"
cd "$HOME" || exit 1
/bin/cat "$HERE/Welcome.txt"
exec /bin/zsh -f
""")
        (resources / "Start.command").chmod(0o755)
        (resources / "Welcome.txt").write_text(
            f"\n{lock['appName']} {version}\n\n{lock['intro']}\n\n"
        )
        run(
            [
                "/usr/bin/clang",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-arch",
                "arm64",
                f"-mmacosx-version-min={lock['minimumMacOS']}",
                str(HERE / "launcher.c"),
                "-o",
                str(macos / "Launcher"),
            ]
        )
        for name in project["scripts"]:
            shutil.copyfile(macos / "Launcher", macos / name)
            (macos / name).chmod(0o755)
        info = {
            "CFBundleName": lock["appName"],
            "CFBundleDisplayName": lock["appName"],
            "CFBundleIdentifier": lock["bundleId"],
            "CFBundleExecutable": "Launcher",
            "CFBundlePackageType": "APPL",
            "CFBundleVersion": version,
            "CFBundleShortVersionString": version,
            "LSMinimumSystemVersion": lock["minimumMacOS"],
            "LSUIElement": True,
            "NSHighResolutionCapable": True,
        }
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
        verify(app, lock)
        # Ad-hoc signatures only. Distribution signing/notarization is a separate release step.
        run(["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app)])
        run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)])
        stem = f"{lock['appName']}-{version}-arm64"
        final = args.output / app.name
        if final.exists():
            shutil.rmtree(final)
        app.rename(final)
        zip_path = args.output / f"{stem}.zip"
        run(
            [
                "/usr/bin/ditto",
                "-c",
                "-k",
                "--sequesterRsrc",
                "--keepParent",
                str(final),
                str(zip_path),
            ]
        )
        image_root = work / "image"
        image_root.mkdir()
        run(["/usr/bin/ditto", str(final), str(image_root / final.name)])
        (image_root / "Applications").symlink_to("/Applications")
        shutil.copyfile(HERE / "README.md", image_root / "README.md")
        dmg = args.output / f"{stem}.dmg"
        run(
            [
                "/usr/bin/hdiutil",
                "create",
                "-ov",
                "-format",
                "UDZO",
                "-volname",
                lock["appName"],
                "-srcfolder",
                str(image_root),
                str(dmg),
            ]
        )
        (args.output / f"{stem}-SHA256SUMS.txt").write_text(
            "".join(f"{digest(p)}  {p.name}\n" for p in (dmg, zip_path))
        )
        print(f"Built {dmg}\nBuilt {zip_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", default="HEAD")
    parser.add_argument("--app-wheel", type=Path)
    parser.add_argument("--native-prefix", type=Path, default=ROOT / "third_party")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/macos-package")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/macos")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Build and verify on an Apple Silicon Mac")
    lock = json.loads(LOCK.read_text())
    if args.verify:
        verify(args.verify.resolve(), lock)
    else:
        if lock.get("nativeLibraries") and not args.app_wheel:
            parser.error(
                "QC requires --app-wheel from the selected source revision; see README.md"
            )
        build(args, lock)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Package command failed:\n{error.stdout}\n{error.stderr}"
        ) from error
