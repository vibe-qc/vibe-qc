#!/usr/bin/env python3
"""Observe FFTW in a final prefix; never qualify provenance or admit reuse.

This standalone bootstrap probe needs no installed vibeqc package. Darwin is
the only supported isolation backend. A caller must separately authenticate
build inputs, install manifests and the candidate's immutable identity.
"""
import ctypes
import errno
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

# -I excludes ambient imports and the script directory. Add only our trusted
# tracked helper directory, never the candidate's package or environment.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attest_native_image import file_digest, loaded_symbol_path


def canonical(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("paths must be absolute, existing and free of aliases")
    return path


def contained_file(root, relative):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("expected a relative candidate file")
    path = canonical(root / relative)
    path.relative_to(root)
    if not path.is_file():
        raise ValueError("candidate file is not regular")
    return path


def header_constants(data):
    """Read only the two simple public ABI constants used by this operation."""
    patterns = {
        "forward": r"^\s*#\s*define\s+FFTW_FORWARD\s+\(?(-1)\)?\s*$",
        "estimate": r"^\s*#\s*define\s+FFTW_ESTIMATE\s+\(1U?\s*<<\s*(6)\)\s*$",
    }
    values = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, data.decode("ascii"), re.MULTILINE)
        if len(matches) != 1:
            raise ValueError("unsupported or ambiguous FFTW header constant: " + name)
        values[name] = int(matches[0]) if name == "forward" else 1 << int(matches[0])
    return values


def require_denial(donor):
    # The parent already proved this directory exists and is readable. ENOENT
    # must never be mistaken for isolation, including a vanished donor.
    try:
        with os.scandir(donor) as entries:
            next(entries, None)
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EPERM):
            return
        raise ValueError("donor denial not demonstrated: " + str(error)) from error
    raise ValueError("donor prefix remains readable")


def native_operation(library_path, constants):
    library = ctypes.CDLL(str(library_path))
    plan = library.fftw_plan_dft_1d
    execute = library.fftw_execute
    destroy = library.fftw_destroy_plan
    for function in (plan, execute, destroy):
        observed = loaded_symbol_path(ctypes.cast(function, ctypes.c_void_p).value)
        if observed != library_path:
            raise ValueError("FFTW symbol resolved outside candidate library")
    pair = ctypes.c_double * 2
    array = pair * 4
    source = array(*(pair(float(i), 0.0) for i in range(1, 5)))
    target = array()
    plan.argtypes = [ctypes.c_int, ctypes.POINTER(pair), ctypes.POINTER(pair),
                     ctypes.c_int, ctypes.c_uint]
    plan.restype = ctypes.c_void_p
    execute.argtypes = [ctypes.c_void_p]
    execute.restype = None
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = None
    handle = plan(4, source, target, constants["forward"], constants["estimate"])
    if not handle:
        raise ValueError("FFTW refused the four-point plan")
    try:
        execute(handle)
        result = [[float(x[0]), float(x[1])] for x in target]
    finally:
        destroy(handle)
    expected = [[10.0, 0.0], [-2.0, 2.0], [-2.0, 0.0], [-2.0, -2.0]]
    if any(not math.isfinite(a) or abs(a - b) > 1e-12
           for row, reference in zip(result, expected) for a, b in zip(row, reference)):
        raise ValueError("FFTW four-point transform failed")
    return result


def child_probe(request):
    require_denial(request["donor"])
    root = canonical(request["candidate"])
    library = contained_file(root, request["library"])
    header = contained_file(root, "include/fftw3.h")
    before = {str(path): file_digest(path) for path in (library, header)}
    # Open the actual candidate header and use its constants in the native call.
    # This is header ABI data, not a chemistry basis-data qualification.
    constants = header_constants(header.read_bytes())
    result = native_operation(library, constants)
    if any(file_digest(Path(path)) != digest for path, digest in before.items()):
        raise ValueError("candidate bytes changed during probe")
    return {"schema": "vibeqc.fftw-prefix-probe/1", "decision": "observed",
            "isolation": "denied-donor-prefix", "operation": "fftw-four-point-dft",
            "result": result, "constants": constants,
            "libraries": [{"path": str(library), "sha256": before[str(library)]}],
            "data": [{"path": str(header), "sha256": before[str(header)]}],
            "data_scope": "FFTW header ABI constants",
            "source_build_provenance": "not_attested", "seed_admission": "not_evaluated"}


def run_probe(request):
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise ValueError("donor-read isolation unsupported on this platform")
    if set(request) != {"candidate", "donor", "library"}:
        raise ValueError("expected exactly candidate, donor and library")
    root, donor = canonical(request["candidate"]), canonical(request["donor"])
    if not root.is_dir() or not donor.is_dir():
        raise ValueError("candidate and donor must be directories")
    if root == donor or root in donor.parents or donor in root.parents:
        raise ValueError("candidate and donor must be disjoint")
    with os.scandir(donor) as entries:
        next(entries, None)
    library = contained_file(root, request["library"])
    header = contained_file(root, "include/fftw3.h")
    before = {str(path): file_digest(path) for path in (library, header)}
    # Use sandbox parameters, not interpolation of filesystem paths into SBPL.
    profile = '(version 1)(allow default)(deny file-read* (subpath (param "DONOR")))'
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith(("DYLD_", "LD_", "PYTHON"))}
    completed = subprocess.run(
        ["/usr/bin/sandbox-exec", "-D", "DONOR=" + str(donor), "-p", profile,
         sys.executable, "-I", str(Path(__file__).resolve()), "--isolated-child"],
        input=json.dumps(request), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=30, env=environment, check=False)
    if completed.returncode:
        raise ValueError("isolated probe failed: " + completed.stderr.strip()[:2000])
    response = json.loads(completed.stdout)
    if response.get("decision") != "observed":
        raise ValueError("isolated probe did not provide observations")
    if any(file_digest(Path(path)) != digest for path, digest in before.items()):
        raise ValueError("candidate bytes changed across isolated probe")
    with os.scandir(donor) as entries:
        next(entries, None)
    return response


def main():
    try:
        request = json.load(sys.stdin)
        if sys.argv[1:] == ["--isolated-child"]:
            result = child_probe(request)
        elif not sys.argv[1:]:
            result = run_probe(request)
        else:
            raise ValueError("supply the probe request on stdin")
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            subprocess.SubprocessError) as error:
        print(json.dumps({"schema": "vibeqc.fftw-prefix-probe/1",
                          "decision": "hold", "reason": str(error)}))
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
