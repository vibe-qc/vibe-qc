"""ECP XML library bundled inside the Python package.

The ECP XML library (libecpint's ecp10mdf / ecp28mdf / ecp46mdf /
ecp60mdf / ecp78mdf / lanl2dz) lives at
``python/vibeqc/ecp_library/xml/`` so a fresh ``pip install -e .``
from the vibe-qc checkout ships everything ``compute_ecp_matrix``
needs — no LIBECPINT_SHARE_DIR fiddling, no Homebrew dependency.

These tests pin the contract:

  1. The bundled directory exists and contains all expected XML files.
  2. ``vibeqc.__init__`` resolves the share-dir to that bundled path.
  3. ``$VIBEQC_ECP_SHARE_DIR`` is set so the C++ ``compute_ecp_matrix``
     path (called from ``run_rhf`` etc. with an empty ``share_dir``)
     reads the right location.
  4. ``vq.compute_ecp_matrix(...)`` with no explicit ``share_dir``
     succeeds against the bundled library.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import vibeqc as _vq
import vibeqc as vq


BUNDLED_ECP_DIR = (
    Path(_vq.__file__).resolve().parent / "ecp_library"
)


def test_bundled_xml_directory_exists():
    """The wheel must ship the XML library — without it, every ECP
    SCF call would fail with "stoi: no conversion" when libecpint
    can't find the file."""
    assert BUNDLED_ECP_DIR.is_dir(), (
        f"Bundled ECP library missing at {BUNDLED_ECP_DIR}; the wheel "
        "does not ship the XML files."
    )
    expected = {
        "ecp10mdf.xml", "ecp28mdf.xml", "ecp46mdf.xml",
        "ecp60mdf.xml", "ecp78mdf.xml", "lanl2dz.xml",
    }
    actual = {p.name for p in (BUNDLED_ECP_DIR / "xml").iterdir()
              if p.suffix == ".xml"}
    missing = expected - actual
    assert not missing, f"Missing bundled XML files: {sorted(missing)}"


def test_share_dir_resolved_at_import():
    """``vibeqc/__init__.py`` resolves the share-dir to the bundled
    location. The string is the path containing ``xml/`` (libecpint
    appends ``"/xml/" + name + ".xml"`` itself; api.cpp:73)."""
    assert _vq._VIBEQC_ECP_SHARE_DIR == str(BUNDLED_ECP_DIR)


def test_env_var_set_for_cpp_paths():
    """``run_rhf(..., ecp_centers=...)`` calls compute_ecp_matrix
    from C++ with an empty share_dir, falling back to
    ``$VIBEQC_ECP_SHARE_DIR``. We set that env var from
    __init__ so the fallback resolves to the bundled dir."""
    actual = os.environ.get("VIBEQC_ECP_SHARE_DIR")
    assert actual == str(BUNDLED_ECP_DIR), (
        f"VIBEQC_ECP_SHARE_DIR={actual!r}, expected "
        f"{BUNDLED_ECP_DIR}"
    )


def test_compute_ecp_matrix_uses_bundled_library_by_default():
    """With share_dir omitted, the bundled path must be the one
    libecpint reads — verified by a successful ECP integral on Zn."""
    mol = vq.Molecule([vq.Atom(30, [0.0, 0.0, 0.0])], 0, 1)
    basis = vq.BasisSet(mol, "6-31g")
    ecp = vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])
    V = vq.compute_ecp_matrix(basis, [ecp], "ecp10mdf")
    nbf = basis.nbasis
    assert V.shape == (nbf, nbf)
    # Non-trivial Stuttgart-Köln 10-core ECP on Zn produces a
    # well-populated matrix; pin the Frobenius norm rather than
    # individual elements.
    import numpy as np
    assert np.linalg.norm(V) > 1.0


def test_lanl2dz_also_in_bundle():
    """Pin LANL2DZ specifically — second supported library, separate
    XML file, ensures the bundling caught more than just ecp10mdf."""
    mol = vq.Molecule([vq.Atom(30, [0.0, 0.0, 0.0])], 0, 1)
    basis = vq.BasisSet(mol, "6-31g")
    ecp = vq.ECPCenter(Z=30, xyz=[0.0, 0.0, 0.0])
    V = vq.compute_ecp_matrix(basis, [ecp], "lanl2dz")
    import numpy as np
    assert np.linalg.norm(V) > 1.0
