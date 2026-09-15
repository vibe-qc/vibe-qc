"""The two basis-name canonicalisers must agree.

``vibeqc.basis_registry.canonical_basis_name`` resolves a spelling for
*metadata*; the C++ ``basis_data_name`` resolves it for the *file* libint
opens. They were independent, so a spelling the registry knew could still fail
in ``BasisSet``: ``sto3g`` was an alias here and a hard error there (#743).

These tests pin the agreement in the direction that matters -- every alias the
registry declares must actually construct -- so adding a row to
``registry.toml`` without the matching line in ``cpp/src/basis.cpp`` fails
here rather than drifting silently.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import vibeqc as vq
from vibeqc.basis_registry import canonical_basis_name

_REGISTRY = (
    Path(vq.__file__).parent / "basis_library" / "registry.toml"
)


def _aliases() -> dict[str, str]:
    with _REGISTRY.open("rb") as handle:
        return tomllib.load(handle).get("aliases", {})


def _water():
    a = 1.8
    return vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, a]), vq.Atom(1, [0.0, a, 0.0])],
        0,
        1,
    )


def test_registry_declares_aliases():
    """A guard on the guard: an empty table would make the rest vacuous."""
    assert len(_aliases()) >= 6


@pytest.mark.parametrize("spelling", sorted(_aliases()))
def test_every_registry_alias_constructs_a_basis(spelling):
    """The C++ name resolution honours every alias the registry declares."""
    basis = vq.BasisSet(_water(), spelling)
    assert basis.nbasis > 0


@pytest.mark.parametrize("spelling", sorted(_aliases()))
def test_alias_and_target_are_the_same_basis(spelling):
    """Resolving through the alias gives the target basis, not a near-miss."""
    target = _aliases()[spelling]
    assert canonical_basis_name(spelling) == target
    assert vq.BasisSet(_water(), spelling).nbasis == vq.BasisSet(_water(), target).nbasis


@pytest.mark.parametrize(
    "spelling",
    ["STO3G", "sto3g", "6-31G(d)", "6-31g(d,p)", "6-311+G(3df,2p)"],
)
def test_case_and_punctuation_spellings_resolve(spelling):
    """Case and Gaussian-style punctuation reach the same file."""
    assert vq.BasisSet(_water(), spelling).nbasis > 0


def test_an_unknown_name_still_fails_clearly():
    """Alias resolution must not turn a typo into a silent fallback."""
    with pytest.raises(RuntimeError, match="no shells loaded|has no functions"):
        vq.BasisSet(_water(), "not-a-real-basis")
