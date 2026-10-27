#!/usr/bin/env python3
"""Authenticate a native-library identity symbol against its requested loaded image.

Metadata is not binary identity. Each probe runs in a bounded child with loader
search overrides removed, and dladdr binds the symbol to the requested real
path. Before/after digests detect changes during the probe. This checks runtime
identity, not source/build-option provenance or adversarial filesystem races.
Keep this standalone bootstrap helper compatible with Python 3.6 and newer;
source preparation may run before the project Python environment exists.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List


class DlInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("base", ctypes.c_void_p),
                ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]


def artifact_paths(install: Path, dependency: str) -> List[Path]:
    basename = {"fftw": "libfftw3", "libxc": "libxc", "spglib": "libsymspg", "openblas": "libopenblas", "libint": "libint2"}[dependency]
    install = install.resolve(strict=True)
    found = set()
    for directory in (install / "lib", install / "lib64"):
        for pattern in (f"{basename}.so*", f"{basename}.dylib", f"{basename}.*.dylib"):
            for item in directory.glob(pattern):
                artifact = item.resolve(strict=True)
                try:
                    artifact.relative_to(install)
                except ValueError:
                    raise ValueError(f"{dependency} artifact escapes install: {item}")
                if not artifact.is_file():
                    raise ValueError(f"{dependency} artifact escapes install or is not a file: {item}")
                found.add(artifact)
    if not found:
        raise ValueError(f"no primary {dependency} shared library under {install}")
    if dependency == "libint" and len(found) != 1:
        raise ValueError("ambiguous Libint shared images; select one actual C library")
    return sorted(found)


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def parse_version(value: str) -> str:
    match = re.fullmatch(r"fftw-(\d+\.\d+\.\d+)(?:-[A-Za-z0-9._+-]+)?", value)
    if not match:
        raise ValueError(f"unrecognized FFTW exported version: {value!r}")
    return match.group(1)


def parse_openblas_configuration(configuration: str) -> str:
    match = re.match(r"OpenBLAS (\d+\.\d+\.\d+[A-Za-z0-9._+-]*)(?: |$)", configuration)
    if not match:
        raise ValueError(f"unrecognized OpenBLAS runtime configuration: {configuration!r}")
    # Eigen's non-MKL BLAS declarations use int indices. A same-symbol ILP64
    # library is not ABI-compatible; thread/kernel flags are merely reported.
    if "USE64BITINT" in configuration.split():
        raise ValueError("OpenBLAS ILP64 (USE64BITINT) is incompatible with the vendored LP64 backend")
    if {"NO_LAPACK", "NO_LAPACKE"}.intersection(configuration.split()):
        raise ValueError("OpenBLAS runtime configuration disables the required LAPACK/LAPACKE backend")
    return match.group(1)


def loaded_symbol_path(address: int) -> Path:
    # Darwin exports dladdr from libSystem, Linux from libc or libdl.
    if sys.platform not in ("darwin", "linux"):
        raise ValueError(f"loaded-image attestation unsupported on {sys.platform}")
    system = ctypes.CDLL(None)
    try:
        dladdr = system.dladdr
    except AttributeError:
        if sys.platform != "linux":
            raise ValueError("dladdr unavailable")
        dladdr = ctypes.CDLL("libdl.so.2").dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    info = DlInfo()
    if dladdr(address, ctypes.byref(info)) == 0 or not info.name:
        raise ValueError("cannot identify the image owning the version entry point")
    return Path(os.fsdecode(info.name)).resolve(strict=True)


def probe_identity(path: str, dependency: str) -> Dict[str, str]:
    requested = Path(path).resolve(strict=True)
    before = file_digest(requested)
    library = ctypes.CDLL(str(requested))
    if dependency == "fftw":
        symbol = ctypes.c_char.in_dll(library, "fftw_version")
        address = ctypes.addressof(symbol)
    elif dependency == "libxc":
        # Public libxc API: void xc_version(int *major, int *minor, int *micro).
        # Bind the function itself before calling it; a returned string alone
        # cannot establish which image supplied the entry point.
        symbol = library.xc_version
        symbol.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
        symbol.restype = None
        address = ctypes.cast(symbol, ctypes.c_void_p).value
    elif dependency == "spglib":
        # Public spglib API: const char *spg_get_version(void).
        symbol = library.spg_get_version
        symbol.argtypes = []
        symbol.restype = ctypes.c_char_p
        address = ctypes.cast(symbol, ctypes.c_void_p).value
    elif dependency == "libint":
        # libint2_iface.h declares void libint2_static_init(void). Bind it,
        # but never call initialization, cleanup or integral kernels.
        symbol = library.libint2_static_init
        address = ctypes.cast(symbol, ctypes.c_void_p).value
    else:
        # Public cblas.h API: char *openblas_get_config(void).
        symbol = library.openblas_get_config
        symbol.argtypes = []
        symbol.restype = ctypes.c_char_p
        address = ctypes.cast(symbol, ctypes.c_void_p).value
    loaded = loaded_symbol_path(address)
    if loaded != requested:
        raise ValueError(f"{dependency} loader substitution: requested {requested}, loaded {loaded}")
    if dependency == "fftw":
        version = parse_version(ctypes.string_at(address).decode("ascii"))
    elif dependency == "libxc":
        parts = [ctypes.c_int() for _ in range(3)]
        symbol(*(ctypes.byref(part) for part in parts))
        if any(part.value < 0 or part.value > 9999 for part in parts):
            raise ValueError("unrecognized libxc exported version")
        version = ".".join(str(part.value) for part in parts)
    elif dependency == "spglib":
        exported = symbol()
        if exported is None:
            raise ValueError("spglib returned no exported version")
        version = exported.decode("ascii")
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9._+-]+)?", version):
            raise ValueError(f"unrecognized spglib exported version: {version!r}")
    elif dependency == "libint":
        cleanup = library.libint2_cleanup_eri
        # libint2_build_eri is an array of function pointers, not a function.
        # Address one byte of its storage without dereferencing any entry.
        table = ctypes.c_char.in_dll(library, "libint2_build_eri")
        for name, entry_address in (
                ("libint2_cleanup_eri", ctypes.cast(cleanup, ctypes.c_void_p).value),
                ("libint2_build_eri", ctypes.addressof(table))):
            owner = loaded_symbol_path(entry_address)
            if owner != requested:
                raise ValueError(f"Libint symbol {name} belongs to {owner}, not {requested}")
        # No safe runtime version/capability query is exposed by this C API.
        version = "not_attested"
    else:
        # The builder promises one shared BLAS/LAPACK/LAPACKE backend. Inspect
        # required exports without executing any numerical kernel; dlopen can
        # otherwise satisfy a missing export from a dependency image.
        for name in ("cblas_dgemm", "dgemm_", "dsyev_", "dpotrf_",
                     "LAPACKE_dsyev", "LAPACKE_dpotrf"):
            entry = getattr(library, name)
            owner = loaded_symbol_path(ctypes.cast(entry, ctypes.c_void_p).value)
            if owner != requested:
                raise ValueError(f"OpenBLAS backend export {name} belongs to {owner}, not {requested}")
        exported = symbol()
        if exported is None:
            raise ValueError("OpenBLAS returned no runtime configuration")
        configuration = exported.decode("ascii")
        version = parse_openblas_configuration(configuration)
    after = file_digest(requested)
    if before != after:
        raise ValueError(f"{dependency} artifact changed during identity probe: {requested}")
    record = {"version": version, "requested_path": str(requested),
              "loaded_path": str(loaded), "sha256": after}
    if dependency == "libint":
        record.update(runtime_capabilities="not_attested", identification="uncalled C API symbols")
    if dependency == "openblas":
        record.update(runtime_configuration=configuration, integer_abi="LP64")
    return record


def attest_install(install: Path, dependency: str) -> List[Dict[str, str]]:
    paths = artifact_paths(install, dependency)
    records = []
    # Environment inherited by dlopen must not substitute a same-basename image.
    # dladdr checks binding independently, including direct --attest invocations.
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith(("DYLD_", "LD_"))}
    for path in paths:
        before = file_digest(path)
        probe = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), dependency, "--attest", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
            timeout=15, check=False, env=environment,
        )
        if probe.returncode:
            raise ValueError(f"cannot authenticate {dependency} artifact {path}: {probe.stderr.strip()}")
        actual = json.loads(probe.stdout)
        if not isinstance(actual, dict) or any(actual.get(k) != str(path)
                for k in ("requested_path", "loaded_path")):
            raise ValueError(f"{dependency} probe did not bind the requested artifact: {path}")
        if actual.get("sha256") != before or file_digest(path) != before:
            raise ValueError(f"{dependency} probe digest mismatch: {path}")
        records.append(actual)
    return records


def main(argv: List[str]) -> int:
    dependency = argv[0] if argv else "native library"
    try:
        if dependency not in ("fftw", "libxc", "spglib", "openblas", "libint"):
            raise ValueError("usage: attest_native_image.py {fftw|libxc|spglib|openblas|libint} INSTALL")
        args = argv[1:]
        if len(args) == 2 and args[0] == "--attest":
            print(json.dumps(probe_identity(args[1], dependency)))
        elif len(args) == 1:
            print(json.dumps({"artifacts": attest_install(Path(args[0]), dependency),
                "source_build_provenance": "not_attested",
                "interpretation": "Exported runtime version is not the upstream source release."}))
        else:
            raise ValueError("usage: attest_native_image.py {fftw|libxc|spglib|openblas|libint} INSTALL")
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as error:
        print(f"{dependency} artifact identity unavailable: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
