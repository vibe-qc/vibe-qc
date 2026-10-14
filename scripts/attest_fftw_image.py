#!/usr/bin/env python3
"""Compatibility entry point for FFTW loaded-image admission."""
import sys

from attest_native_image import main

if __name__ == "__main__":
    raise SystemExit(main(["fftw", *sys.argv[1:]]))
