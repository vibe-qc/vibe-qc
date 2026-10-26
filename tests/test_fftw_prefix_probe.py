"""Probe controls without compiling dependencies or qualifying historical data."""
import errno
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_fftw_prefix.py"
spec = importlib.util.spec_from_file_location("fftw_prefix_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_constants_are_read_and_unknown_or_duplicate_rejected():
    header = b"#define FFTW_FORWARD (-1)\n#define FFTW_ESTIMATE (1U << 6)\n"
    assert probe.header_constants(header) == {"forward": -1, "estimate": 64}
    for bad in (header.replace(b"(-1)", b"(+1)"), header + header, b""):
        with pytest.raises(ValueError):
            probe.header_constants(bad)


def test_missing_or_readable_donor_never_counts_as_denial(tmp_path):
    with pytest.raises(ValueError, match="remains readable"):
        probe.require_denial(tmp_path)
    with pytest.raises(ValueError, match="not demonstrated"):
        probe.require_denial(tmp_path / "missing")
    for denied in (errno.EACCES, errno.EPERM):
        with patch.object(probe.os, "scandir", side_effect=OSError(denied, "denied")):
            probe.require_denial(tmp_path)


def test_paths_cannot_escape_or_alias(tmp_path):
    root = tmp_path.resolve()
    (root / "real").write_bytes(b"data")
    (root / "link").symlink_to(root / "real")
    for relative in ("../real", str(root / "real"), "link"):
        with pytest.raises(ValueError):
            probe.contained_file(root, relative)


def test_unsupported_isolation_is_hold():
    with patch.object(sys, "platform", "linux"):
        with pytest.raises(ValueError, match="isolation unsupported"):
            probe.run_probe({})


def test_child_requires_denial_before_native_call(tmp_path):
    with patch.object(probe, "native_operation") as operation:
        with pytest.raises(ValueError, match="remains readable"):
            probe.child_probe({"donor": str(tmp_path)})
        operation.assert_not_called()


def test_child_refuses_changed_candidate_bytes(tmp_path):
    root = tmp_path.resolve()
    (root / "include").mkdir()
    header = root / "include/fftw3.h"
    header.write_text("#define FFTW_FORWARD (-1)\n#define FFTW_ESTIMATE (1U << 6)\n")
    (root / "library").write_bytes(b"fixture only; never loaded")
    def mutation(*args):
        header.write_text("changed")
        return []
    with patch.object(probe, "require_denial"), patch.object(probe, "native_operation", side_effect=mutation):
        with pytest.raises(ValueError, match="bytes changed"):
            probe.child_probe({"donor": "unused", "candidate": str(root), "library": "library"})


def test_isolation_backend_really_denies_reads(tmp_path):
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        pytest.skip("Darwin sandbox backend only")
    donor = tmp_path.resolve() / "donor"
    donor.mkdir()
    secret = donor / "data"
    secret.write_text("must not be read")
    profile = '(version 1)(allow default)(deny file-read* (subpath (param "DONOR")))'
    code = '''import errno, pathlib, sys
try:
    pathlib.Path(sys.argv[1]).read_bytes()
except OSError as error:
    assert error.errno in (errno.EACCES, errno.EPERM), error
else:
    raise AssertionError("donor read succeeded")
'''
    result = probe.subprocess.run(
        ["/usr/bin/sandbox-exec", "-D", "DONOR=" + str(donor), "-p", profile,
         sys.executable, "-I", "-c", code, str(secret)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert secret.read_text() == "must not be read"


def test_parent_rejects_overlapping_prefixes(tmp_path):
    with patch.object(sys, "platform", "darwin"), patch.object(Path, "is_file", return_value=True):
        with pytest.raises(ValueError, match="disjoint"):
            probe.run_probe({"candidate": str(tmp_path.resolve()),
                             "donor": str(tmp_path.resolve()), "library": "fake"})
