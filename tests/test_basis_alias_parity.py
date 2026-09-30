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
  ``BasisSet`` accepts.
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
