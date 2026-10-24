#!/usr/bin/env python3
"""Apply the bounded spin-fraction fix to the pinned Libxc 7.0.0 source.

PBE defines zeta=(n_up-n_down)/n and phi from (1 +/- zeta)**(2/3)
(Perdew et al., DOI 10.1103/PhysRevLett.77.3865, p. 3866, after Eq. 3).
Evaluate those fractions as 2*n_spin/n to avoid cancellation at full spin
polarization. The same issue occurs in r2SCAN correlation (Furness et al.,
DOI 10.1021/acs.jpclett.0c02405); its regularization is left unchanged.

Only the two generated correlation kernels change, at all five derivative
orders. No density threshold, spin cutoff, parameter or derivative formula
changes. Upstream copyright and MPL-2.0 notices remain intact. Exact input
hashes prevent this patch from overwriting an upstream or local change.
"""

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from attest_native_image import artifact_paths, file_digest, loaded_symbol_path


VERSION = b"stable-spin-fractions-v1"
MARKER = '''
/* vibe-qc: identifies the pinned stable-spin-fraction build. */
const char *vibeqc_libxc_spin_fractions_version(void)
{
  return "stable-spin-fractions-v1";
}
'''
TARGETS = (
    ("src/maple2c/gga_exc/gga_c_pbe.c",
     "0b0045c5027af1e250128a4243285715714d31ebbf6ff83dab52ca03cc17ad70",
     (("  t43 = 0.1e1 + t42;", "  t43 = 2.0 * (rho[0] / t7); /* stable spin fraction */"),
      ("  t50 = 0.1e1 - t42;", "  t50 = 2.0 * (rho[1] / t7); /* stable spin fraction */"))),
    ("src/maple2c/mgga_exc/mgga_c_r2scan.c",
     "a61ab7d2dbfa3e99ea7c9a8ff18182344b9ce727af96b385804eb9cd65f7ce1d",
     (("  t44 = 0.1e1 + t43;", "  t44 = 2.0 * (rho[0] / t8); /* stable spin fraction */"),
      ("  t51 = 0.1e1 - t43;", "  t51 = 2.0 * (rho[1] / t8); /* stable spin fraction */"))),
    ("src/gga_c_pbe.c",
     "d449da1e327e5c5c17dc50174c25793aeea1876c4d1250d53ae12ded3e6395da", ()),
)


def patched_text(text: str, digest: str, replacements: tuple) -> str:
    # Reverse only our exact edits to recognize an already patched file.
    original = text
    if replacements:
        for old, new in replacements:
            original = original.replace(new + "\n", old + "\n")
    elif original.endswith(MARKER):
        original = original[:-len(MARKER)]
    if hashlib.sha256(original.encode()).hexdigest() != digest:
        raise ValueError("source differs from pinned Libxc 7.0.0; refusing to overwrite")
    result = original
    for old, new in replacements:
        if result.count(old + "\n") != 5:
            raise ValueError("expected one spin fraction per derivative order")
        result = result.replace(old + "\n", new + "\n")
    return result if replacements else result + MARKER


def patch_source(source: Path) -> None:
    pending = []
    # Validate every target before touching any file.
    for relative, digest, replacements in TARGETS:
        path = source / relative
        try:
            before = path.read_text()
            after = patched_text(before, digest, replacements)
        except ValueError as error:
            raise ValueError(f"{relative}: {error}") from error
        if before != after:
            pending.append((path, after))
    for path, after in pending:
        path.write_text(after)
    print(f"Libxc stable spin fractions: {len(pending)} source files updated")


def probe_image(path: Path) -> None:
    path = path.resolve(strict=True)
    before = file_digest(path)
    library = ctypes.CDLL(str(path))
    marker = library.vibeqc_libxc_spin_fractions_version
    marker.argtypes = []
    marker.restype = ctypes.c_char_p
    if loaded_symbol_path(ctypes.cast(marker, ctypes.c_void_p).value) != path:
        raise ValueError("spin-fraction marker belongs to another image")
    if marker() != VERSION or file_digest(path) != before:
        raise ValueError("missing stable spin fractions or image changed during probe")


def check_install(install: Path) -> None:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LD_", "DYLD_"))}
    for path in artifact_paths(install, "libxc"):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe-image", str(path)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30,
        )
        if result.returncode:
            raise ValueError("installed Libxc lacks the stable-spin-fraction patch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source", type=Path)
    modes.add_argument("--check-install", type=Path)
    modes.add_argument("--probe-image", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.source:
            patch_source(args.source)
        elif args.check_install:
            check_install(args.check_install)
        else:
            probe_image(args.probe_image)
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired) as error:
        print(f"Libxc spin-fraction check: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
