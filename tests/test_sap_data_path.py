"""SAP atomic-potential tables resolve at RUNTIME, not from the build tree.

The SAP ``.g94`` location used to come solely from libint's *compile-time*
``LIBINT_DATADIR`` macro, so ``initial_guess=SAP`` -- and ``AUTO``, which
``GuessEngine::resolve_auto`` maps to SAP for molecular closed-shell --
died with ``SAP: cannot open atomic-potential file <build-path>/basis/
sap_helfem_large.g94`` on every install whose build directory no longer
exists at run time: relocatable bundles, wheels installed elsewhere,
multi-stage containers. Ordinary basis loading never had this problem
because libint honours ``$LIBINT_DATA_PATH`` (``cpp/src/basis.cpp``).

Resolution now mirrors ``libint2::BasisSet::data_path()``:
``$LIBINT_DATA_PATH`` first, the compile-time value only as a fallback --
so SAP tables are found exactly where ordinary basis sets are, and a
genuinely missing table reports every location that was searched.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import compute_sap_potential_molecular

SAP_NAME = "sap_helfem_large"


def _h2():
    """H2 / STO-3G -- the smallest system that exercises the SAP table."""
    mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, -0.7]),
                       vq.Atom(1, [0.0, 0.0, 0.7])])
    return mol, vq.BasisSet(mol, "sto-3g")


def _bundled_basis_file(name: str) -> Path:
    """Locate a shipped ``<name>.g94`` for this install."""
    candidates = []
    env = os.environ.get("LIBINT_DATA_PATH")
    if env:
        candidates.append(Path(env) / "basis" / f"{name}.g94")
    package_dir = Path(vq.__file__).resolve().parent
    candidates.append(package_dir / "basis_library" / "basis" / f"{name}.g94")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip(f"no bundled {name}.g94 found; searched {candidates}")


def _relocated_data_dir(tmp_path: Path) -> Path:
    """A self-contained data directory standing in for a moved install.

    Carries the orbital basis the test molecule needs (so ordinary basis
    loading keeps working once ``$LIBINT_DATA_PATH`` points here) but no
    SAP table -- each test supplies whichever SAP payload it is probing.
    """
    root = tmp_path / "relocated-install" / "share" / "libint"
    (root / "basis").mkdir(parents=True)
    shutil.copy(_bundled_basis_file("sto-3g"), root / "basis" / "sto-3g.g94")
    return root


def test_sap_table_loads_from_libint_data_path(tmp_path, monkeypatch):
    """A copy of the table under ``$LIBINT_DATA_PATH`` is usable on its own.

    This is the relocated-bundle shape: the data ships with the install,
    the build tree is gone, and the runtime resolution has to find it.
    """
    mol, basis = _h2()
    reference = np.asarray(compute_sap_potential_molecular(basis, mol, SAP_NAME))

    root = _relocated_data_dir(tmp_path)
    shutil.copy(_bundled_basis_file(SAP_NAME), root / "basis" / f"{SAP_NAME}.g94")
    monkeypatch.setenv("LIBINT_DATA_PATH", str(root))

    mol, basis = _h2()
    relocated = np.asarray(compute_sap_potential_molecular(basis, mol, SAP_NAME))
    # Same bytes in, same matrix out -- bit-identical, not merely close.
    assert np.array_equal(relocated, reference)


def test_sap_table_override_is_actually_read(tmp_path, monkeypatch):
    """``$LIBINT_DATA_PATH`` is *read*, not merely tolerated.

    The override carries a table that only fails if it is genuinely opened:
    a P-type shell, which the SAP parser rejects (SAP expansions are
    S-only). Before the runtime-resolution fix the compile-time build-tree
    copy was read unconditionally, the override was ignored, and this
    raised nothing at all.
    """
    root = _relocated_data_dir(tmp_path)
    (root / "basis" / f"{SAP_NAME}.g94").write_text(
        "! Synthetic SAP table -- the P shell is rejected by the parser.\n"
        "****\n"
        "H 0\n"
        "P 1 1.00\n"
        "      1.0000000              1.0000000\n"
        "****\n"
    )
    monkeypatch.setenv("LIBINT_DATA_PATH", str(root))

    mol, basis = _h2()
    with pytest.raises(RuntimeError, match="only S-type shells"):
        compute_sap_potential_molecular(basis, mol, SAP_NAME)


def test_missing_sap_table_error_names_every_searched_path(tmp_path, monkeypatch):
    """A genuinely absent table reports where it looked, not one dead path."""
    root = _relocated_data_dir(tmp_path)
    monkeypatch.setenv("LIBINT_DATA_PATH", str(root))

    mol, basis = _h2()
    with pytest.raises(RuntimeError) as excinfo:
        compute_sap_potential_molecular(basis, mol, "sap_no_such_table")
    message = str(excinfo.value)

    assert str(root / "basis" / "sap_no_such_table.g94") in message
    # The actionable pointer, so the user does not have to read our source.
    assert "LIBINT_DATA_PATH" in message
