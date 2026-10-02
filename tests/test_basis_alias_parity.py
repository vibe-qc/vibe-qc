"""One basis-name resolver for ``BasisSet`` and the Python registry.

``vibeqc.basis_registry.canonical_basis_name`` resolves a spelling for
*metadata*; ``BasisSet`` resolves it for the *file* libint opens. They used to
be two implementations. First ``sto3g`` was a registry alias and a hard error in
``BasisSet`` (#743). Then the C++ side keyed on punctuation-free
``basis_lookup_key`` while Python only lowercased and looked up an alias table,
so ``631gd``, ``sto_3g`` and ``6-31G(d, p)`` loaded a real basis while the
registry reported family ``unknown`` (#30).

Since #30 both go through the C++ ``vibeqc::basis_data_name``. These tests pin
that for spellings users type, not only for a table's own keys:

* the resolved name is the stem of the file ``BasisSet`` actually loads (shell
  by shell), and the spelling gets that stem's registry record;
* a spelling ``BasisSet`` refuses resolves to nothing registered;
* the SAD guess, which loads the atomic basis by name, accepts every spelling
  ``BasisSet`` accepts;
* every vibe-qc writer of a named basis file stores it under that stem, so the
  lookup finds it on a case-sensitive file system too (#321).
"""

from __future__ import annotations

import pytest
import vibeqc as vq
from vibeqc.basis_registry import (
    basis_file_path,
    canonical_basis_name,
    lookup,
)

# spelling -> the file stem it must load.
SPELLINGS = {
    # The eight spellings the removed registry.toml [aliases] table declared.
    "sto6g": "sto-6g",
    "6-31g(d)": "6-31g*",
    "6-31g(d,p)": "6-31g**",
    "6-311g(d,p)": "6-311g**",
    "6-31+g(d,p)": "6-31+g**",
    "6-311+g(d,p)": "6-311+g**",
    "6-311+g(3df,2p)": "6-311+g3df2p",
    "sto3g": "sto-3g",
    # The seven the #30 verifier found loading in C++ but "unknown" in Python.
    "6-31G(d, p)": "6-31g**",
    "6-31G (d)": "6-31g*",
    "631gd": "6-31g*",
    "6311+g(3df,2p)": "6-311+g3df2p",
    "6-311+G(3df, 2p)": "6-311+g3df2p",
    "sto_3g": "sto-3g",
    "Sto 3G": "sto-3g",
    # Case and whitespace.
    "STO3G": "sto-3g",
    "6-31G(D)": "6-31g*",
    "6-311G(d,p)": "6-311g**",
    "  def2-SVP ": "def2-svp",
    "CC-PVDZ": "cc-pvdz",
    "aug-cc-pVDZ-PP": "aug-cc-pvdz-pp",
}

# Targets the registry carries a record or family for. Not every shipped stem
# has one (sto-6g, 6-31+g**, 6-311+g3df2p do not); for those, parity means the
# spelling gets the same record as the stem, which the test also checks.
REGISTERED_TARGETS = {"sto-3g", "6-31g*", "6-31g**", "6-311g**", "def2-svp",
                      "cc-pvdz", "aug-cc-pvdz-pp"}

# BasisSet refuses these, so the registry must not claim them either.
REFUSED = ["not-a-real-basis", "6-311g(d)", "6-31+g(d)", "6-311+g(d)"]


def _mol(zs=(8, 1, 1)):
    atoms = [vq.Atom(z, [0.0, 0.0, 1.8 * k]) for k, z in enumerate(zs)]
    return vq.Molecule(atoms, 0, 1 if sum(zs) % 2 == 0 else 2)


def _signature(basis):
    return [(s.atom_index, s.l, tuple(s.exponents)) for s in basis.shells()]


def test_python_resolver_is_the_cpp_resolver():
    """No second implementation: the registry calls the binding."""
    for spelling in list(SPELLINGS) + REFUSED:
        assert canonical_basis_name(spelling) == vq._vibeqc_core.basis_data_name(
            spelling
        )


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_spelling_resolves_to_the_file_basisset_loads(spelling):
    target = SPELLINGS[spelling]
    assert canonical_basis_name(spelling) == target
    # The same record as the stem itself: family, defaults and ECP rule
    # follow the file that loads, whatever the spelling.
    assert lookup(spelling) == lookup(target)
    if target in REGISTERED_TARGETS:
        assert lookup(spelling).registered, (
            f"{spelling!r} loads {target} but the registry says unknown"
        )
    assert basis_file_path(spelling) == basis_file_path(target) is not None
    mol = _mol((35,)) if target.endswith("-pp") else _mol()
    assert _signature(vq.BasisSet(mol, spelling)) == _signature(vq.BasisSet(mol, target))


@pytest.mark.parametrize("spelling", REFUSED)
def test_refused_spelling_is_refused_by_both(spelling):
    """Resolution must not turn a typo, or a set that does not ship, into a match."""
    assert basis_file_path(spelling) is None
    assert not lookup(spelling).registered
    with pytest.raises(RuntimeError, match="no shells loaded"):
        vq.BasisSet(_mol(), spelling)


def test_basisset_object_resolves_like_its_spelling():
    """``lookup(BasisSet)`` reads the spelling the user gave and resolves it the same way."""
    basis = vq.BasisSet(_mol(), "631gd")
    assert lookup(basis).name == "6-31g*"
    assert lookup(basis).registered


@pytest.mark.parametrize("spelling", ["sto3g", "Sto 3G", "631gd", "6-31G(d, p)", "6-31g(d)"])
def test_sad_guess_accepts_every_spelling_basisset_accepts(spelling):
    """SAD loads its atomic basis by name; before #30 it only lowercased.

    ``6-31G(d)`` and ``sto3g`` then built a molecular basis and failed in SAD
    with "has no functions for Z=8".
    """
    mol = _mol()
    basis = vq.BasisSet(mol, spelling)
    D = vq._vibeqc_core._guess_closed_shell_density(
        mol, basis, 5, vq.InitialGuess.SAD, is_periodic=False
    )
    assert D.shape == (basis.nbasis, basis.nbasis)
    assert float((D * vq.compute_overlap(basis)).sum()) == pytest.approx(10.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Writers store a named basis under the stem the resolver opens (#321)
# ---------------------------------------------------------------------------
#
# Writers that stored the caller's spelling (``dS-an.g94``) produced a file the
# lookup (``ds-an.g94``) could not find on a case-sensitive file system, while a
# case-insensitive one hid it. These tests compare the exact-case directory
# listing with ``canonical_basis_name``, so they fail on either kind of file
# system when a writer bypasses the resolver.

# A mixed-case name, a second one, and an alias spelling the resolver maps to a
# different stem ("sto3g" opens "sto-3g.g94").
WRITER_NAMES = ["dS-an", "Mixed-Name", "sto3g"]


def _pob_h_atoms():
    from pathlib import Path

    from vibeqc.basis_crystal import parse_crystal_atom_basis_file

    source = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP" / "01_H"
    return {"H": parse_crystal_atom_basis_file(source)}


def _h_atom():
    return vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)


def _g94_stems(directory):
    import os

    return sorted(n[: -len(".g94")] for n in os.listdir(directory) if n.endswith(".g94"))


@pytest.mark.parametrize("name", WRITER_NAMES)
def test_temp_library_writes_the_stem_basisset_opens(name):
    from vibeqc.basis_optimization.io import TempBasisLibrary

    with TempBasisLibrary() as lib:
        returned = lib.write_g94(_pob_h_atoms(), basis_name=name)
        assert _g94_stems(lib.path / "basis") == [canonical_basis_name(name)]
        assert returned == canonical_basis_name(name)
        # Both spellings load the written file.
        assert vq.BasisSet(_h_atom(), name).nbasis > 0
        assert vq.BasisSet(_h_atom(), returned).nbasis > 0


@pytest.mark.parametrize("stem", ["Cand", "cand"])
def test_optimizer_candidate_writer_uses_the_resolver_stem(stem):
    pytest.importorskip("vibe_basis")  # calculators.py needs the optional engine API
    from vibeqc.basis_optimization.calculators import _write_basis
    from vibeqc.basis_optimization.io import TempBasisLibrary

    with TempBasisLibrary() as lib:
        name = _write_basis(lib, list(_pob_h_atoms().values()), stem)
        assert _g94_stems(lib.path / "basis") == [canonical_basis_name(name)]
        assert vq.BasisSet(_h_atom(), name).nbasis > 0


def test_filtered_basis_is_written_under_the_resolver_stem(monkeypatch, tmp_path):
    """``filter_basis_by_exponent`` on ``BasisSet(mol, "def2-SVP")`` wrote
    ``_vibeqc_filtered_def2-SVP_<h>.g94`` and then looked up the lower-case
    stem. The filter writes into the active data directory, so the test gives
    it a private one."""
    import shutil

    from vibeqc.basis_filter import filter_basis_by_exponent

    library = tmp_path / "library"
    (library / "basis").mkdir(parents=True)
    shutil.copy2(basis_file_path("def2-svp"), library / "basis" / "def2-svp.g94")
    monkeypatch.setenv("LIBINT_DATA_PATH", str(library))
    monkeypatch.delenv("VIBEQC_FILTERED_BASIS_DIR", raising=False)

    mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])])
    basis = vq.BasisSet(mol, "def2-SVP")
    filtered, report = filter_basis_by_exponent(basis, mol, 0.15)

    written = [s for s in _g94_stems(library / "basis") if s.startswith("_vibeqc_filtered_")]
    assert written == [canonical_basis_name(report.new_name)]
    assert filtered.nbasis < basis.nbasis


def test_inherited_library_file_is_replaced_not_written_through(monkeypatch, tmp_path):
    """``inherit_bundled=True`` hard-links the bundled files into the temporary
    library. Writing a candidate under one of those names must replace the
    link, not truncate the shared inode, or the package's own file changes."""
    import os
    import tempfile

    import vibeqc.basis_library as basis_library
    from vibeqc.basis_optimization.io import TempBasisLibrary

    fake = tmp_path / "fake_library"
    (fake / "basis").mkdir(parents=True)
    bundled = fake / "basis" / "sto-3g.g94"
    original = "! stand-in for a bundled basis file\n"
    bundled.write_text(original)
    monkeypatch.setattr(basis_library, "__file__", str(fake / "__init__.py"), raising=False)
    # Same file system as the fake library, so the inherit step really links.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with TempBasisLibrary(inherit_bundled=True) as lib:
        linked = lib.path / "basis" / "sto-3g.g94"
        assert os.stat(linked).st_ino == os.stat(bundled).st_ino
        lib.write_g94(_pob_h_atoms(), basis_name="sto-3g")
        assert vq.BasisSet(_h_atom(), "sto-3g").nbasis > 0

    assert bundled.read_text() == original
