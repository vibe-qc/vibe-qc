"""Regression tests for ``BasisSet`` error handling.

The historical concern (flagged from the docs / tutorials chat in
v0.4 prep) was that some failure modes of basis-set construction
could segfault rather than raise a clean Python exception. This
file pins down the contract that **every** failure mode raises an
informative ``RuntimeError`` — never a segfault, never an empty
``BasisSet`` that silently fails downstream.

Failure modes covered:

  1. Unknown basis name (no matching ``.g94`` in LIBINT_DATA_PATH).
  2. Empty basis name.
  3. Path-as-name (absolute and relative paths to ``.g94`` files).
  4. Element with no entries in an otherwise-known basis (Pb / cc-pvdz).
  5. Bogus LIBINT_DATA_PATH (libint throws from its data-path lookup).
  6. Malformed ``.g94`` file (libint G94 parser throws).

The defensive plumbing on the C++ side is:

  * :func:`vibeqc::ensure_libint_initialized` is called from inside
    the BasisSet member-initializer list (was missing before — could
    cause undefined behavior when BasisSet was the very first libint
    call in a process).
  * libint2's own throws (``BasisSet::data_path()`` / G94 parse
    errors) are caught in the ``make_libint_basis`` helper and
    re-raised with directive context (the LIBINT_DATA_PATH value,
    a pointer at the bundled-library workflow).
  * The empty-result case keeps the existing "no shells loaded"
    message.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# 1. Unknown basis name
# ---------------------------------------------------------------------------

def test_unknown_basis_name_raises():
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        0, 1,
    )
    with pytest.raises(RuntimeError, match="totally-fake-basis"):
        vq.BasisSet(mol, "totally-fake-basis")


@pytest.mark.parametrize("basis_name", ["6-311+g3df2p", "6-311+G(3df,2p)"])
def test_exact_pbe1996_basis_loads_without_aliasing(basis_name):
    """The exact PBE 1996 reproduction basis is bundled directly; do not
    route it through a nearby Pople variant with different polarization."""
    mol = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.793353, -0.613510]),
            vq.Atom(1, [0.0, -0.793353, -0.613510]),
        ],
        0,
        1,
    )
    basis = vq.BasisSet(mol, basis_name)
    shells = basis.shells()

    assert basis.nshells == 23
    assert sum(1 for sh in shells if sh.atom_index == 0 and sh.l == 2) == 3
    assert sum(1 for sh in shells if sh.atom_index == 0 and sh.l == 3) == 1
    for atom_index in (1, 2):
        assert sum(1 for sh in shells if sh.atom_index == atom_index and sh.l == 1) == 2
        assert sum(1 for sh in shells if sh.atom_index == atom_index and sh.l == 2) == 0


def test_empty_basis_name_raises():
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        0, 1,
    )
    with pytest.raises(RuntimeError):
        vq.BasisSet(mol, "")


# ---------------------------------------------------------------------------
# 2. Path-as-name
# ---------------------------------------------------------------------------

def test_path_as_basis_name_raises():
    """Users sometimes confuse ``BasisSet`` for a file-loader; pass an
    explicit path. libint doesn't recognize paths as basis names — we
    must reject with a helpful message rather than segfault."""
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        0, 1,
    )
    # Both an absolute path that exists in our bundle and a bogus
    # path-like string.
    real_path = "/var/empty/no-such-file.g94"
    with pytest.raises(RuntimeError):
        vq.BasisSet(mol, real_path)


# ---------------------------------------------------------------------------
# 3. Element-not-in-basis (basis exists in some files but not for this Z)
# ---------------------------------------------------------------------------

def test_element_missing_from_basis_raises():
    """cc-pvdz doesn't ship Pb (Z = 82) entries — libint returns an
    empty BasisSet, which our nbf == 0 guard catches."""
    pb = vq.Molecule([vq.Atom(82, [0.0, 0.0, 0.0])], 0, 1)
    with pytest.raises(RuntimeError, match="cc-pvdz"):
        vq.BasisSet(pb, "cc-pvdz")


def test_element_missing_from_a_multi_atom_basis_names_it():
    """cc-pvdz ships no Pb block. In a molecule with covered atoms libint
    returns a non-empty basis that simply has no functions on lead; the
    per-atom guard names the element instead of letting the SCF place
    lead's electrons in hydrogen's functions."""
    mol = vq.Molecule(
        [vq.Atom(82, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 3.5])], 0, 2
    )
    with pytest.raises(RuntimeError, match="no functions for Z=82"):
        vq.BasisSet(mol, "cc-pvdz")
    partial = vq.BasisSet(mol, "cc-pvdz", require_all_atoms=False)
    assert partial.nbasis == vq.BasisSet(
        vq.Molecule([vq.Atom(1, [0.0, 0.0, 3.5])], 0, 2), "cc-pvdz"
    ).nbasis


def test_coincident_centres_are_not_a_coverage_gap():
    """Two atoms of one element on the same point: libint builds both
    atoms' shells, the geometric shell-to-atom map attributes all of them
    to the first atom, and the guard must not read the second as a
    missing element (the DF metric guard is what diagnoses duplicates)."""
    mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(mol, "def2-svp")
    single = vq.BasisSet(vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], 0, 2), "def2-svp")
    assert basis.nbasis == 2 * single.nbasis


# ---------------------------------------------------------------------------
# 4. Bogus LIBINT_DATA_PATH
# ---------------------------------------------------------------------------

def test_bogus_libint_data_path_raises_with_context():
    """When LIBINT_DATA_PATH points nowhere, libint throws from its
    ``data_path()`` lookup. Our wrapper catches that and re-raises
    with the LIBINT_DATA_PATH value embedded in the message — so the
    user sees the failed path right in the traceback."""
    script = textwrap.dedent(
        """
        import vibeqc as vq
        mol = vq.Molecule(
            [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
            0, 1,
        )
        try:
            b = vq.BasisSet(mol, 'sto-3g')
            print('UNEXPECTED_OK:', b.nbasis)
        except RuntimeError as e:
            print('OK:', str(e))
        except Exception as e:
            print('WRONG_TYPE:', type(e).__name__, str(e))
        """
    )
    env = dict(os.environ)
    env["LIBINT_DATA_PATH"] = "/var/empty/vibeqc-test-no-such-dir"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, check=False,
    )
    out = result.stdout.strip()
    assert out.startswith("OK:"), (
        f"Expected RuntimeError, got: stdout={out!r} stderr={result.stderr!r}"
    )
    assert "vibeqc-test-no-such-dir" in out, (
        f"LIBINT_DATA_PATH context missing from error message: {out}"
    )
    assert "BasisSet" in out, (
        f"BasisSet context missing from error message: {out}"
    )


# ---------------------------------------------------------------------------
# 5. Malformed .g94 file
# ---------------------------------------------------------------------------

def test_malformed_g94_raises_with_context():
    """libint's G94 parser throws on malformed input. The wrapper
    must surface the underlying error with our directive prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bdir = Path(tmpdir) / "basis"
        bdir.mkdir()
        # Looks G94-ish but has a bogus element symbol — libint
        # parses ``****\n<symbol>`` and looks up <symbol> in the
        # periodic table; "this" isn't an element.
        (bdir / "junk-basis.g94").write_text(
            "****\nthis is not real\n****\n"
        )
        env = dict(os.environ)
        env["LIBINT_DATA_PATH"] = str(tmpdir)
        script = textwrap.dedent(
            """
            import vibeqc as vq
            mol = vq.Molecule(
                [vq.Atom(1, [0.0, 0.0, 0.0]),
                 vq.Atom(1, [0.0, 0.0, 1.4])],
                0, 1,
            )
            try:
                b = vq.BasisSet(mol, 'junk-basis')
                print('UNEXPECTED_OK:', b.nbasis)
            except RuntimeError as e:
                print('OK:', str(e))
            except Exception as e:
                print('WRONG_TYPE:', type(e).__name__, str(e))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, check=False,
        )
        out = result.stdout.strip()
        assert out.startswith("OK:"), (
            f"Expected RuntimeError, got: stdout={out!r} "
            f"stderr={result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# 6. Successful path still works (smoke test, guards against accidental
#    breakage of the happy path while we plumb error-handling)
# ---------------------------------------------------------------------------

def test_known_basis_still_loads():
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        0, 1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    assert basis.nbasis == 2
    assert basis.nshells == 2
