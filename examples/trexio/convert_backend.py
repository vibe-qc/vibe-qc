#!/usr/bin/env python3
"""Copy all installed-library TREXIO fields to another storage backend.

No SCF calculation or native basis reconstruction is performed. Each state
is a separate file; linked state files must be copied separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vibeqc import read_trexio_fields, write_trexio_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--backend", choices=("hdf5", "text"), required=True)
    args = parser.parse_args()
    fields = read_trexio_fields(args.source)
    # A distinct destination makes this example safe to repeat accidentally.
    path = write_trexio_fields(args.destination, fields, backend=args.backend, overwrite=False)
    print(f"Copied {len(fields)} fields to {path}")


if __name__ == "__main__":
    main()
