"""An ``aug-cc-pV*`` name must load its own shipped file (#30).

libint2 splits any basis name that starts with ``aug-cc-pv`` (except the
``-cabs`` sets) into the parent set plus an ``augmentation-<parent>`` file
(``libint2::BasisSet::decompose_name_into_components``). A missing
augmentation file reads as empty, so before #30 every shipped
``aug-cc-pv*.g94`` without an ``augmentation-`` companion silently loaded
its parent set with no diffuse shells. The affected names were the 4
aug-cc-pVnZ-PP orbital sets, their 4 -PP-RIFIT fits, the 5 all-electron
aug-cc-pVnZ-RIFIT fits and the 4 aug-cc-pV(n+d)Z sets. For example, Br2 RHF
gave bit-identical energies for cc-pVDZ-PP and aug-cc-pVDZ-PP
(-830.965539719857 Ha).

"aug-" means the parent set plus one diffuse function per angular momentum
(Kendall, Dunning, Harrison, J. Chem. Phys. 96, 6796 (1992),
doi:10.1063/1.462569). An aug set therefore always has more shells than its
parent, which the tests below assert.

These tests compare, shell by shell, what ``BasisSet`` actually loads with the
``.g94`` file libint reads for that name. The metadata tests
(test_basis_ecp_sidecars, test_basis_registry) never looked at the loaded
shells, which is how the defect passed a green suite.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import pytest
import vibeqc as vq
from vibeqc.basis_registry import _SYMBOL_TO_Z, basis_file_path

_ZETAS = ("d", "t", "q", "5")

# All 16 cc-pVnZ-PP orbital sets, aug and non-aug, valence and weighted-core.
PP_ORBITAL = [
    f"{aug}cc-p{wc}v{z}z-pp"
    for aug in ("", "aug-")
    for wc in ("", "wc")
    for z in _ZETAS
]
# Their 16 RI fits. The aug-cc-pV*-PP-RIFIT four were affected.
PP_RIFIT = [f"{name}-rifit" for name in PP_ORBITAL]
# All-electron names hit by the same decomposition.
AE_AFFECTED = [f"aug-cc-pv{z}z-rifit" for z in ("d", "t", "q", "5", "6")] + [
    f"aug-cc-pv({z}+d)z" for z in _ZETAS
]


def _library_basis_dir() -> Path:
    return Path(os.environ["LIBINT_DATA_PATH"]) / "basis"


def _shipped_aug_names() -> list[str]:
    """Every ``aug-cc-pv*`` file libint would decompose, found on disk."""
    root = _library_basis_dir()
    return sorted(
        p.stem
        for p in root.glob("aug-cc-pv*.g94")
        if "cabs" not in p.stem
    )


# Gaussian94 angular-momentum letters: S..I are l = 0..6, then J = 7, K = 8,
# L = 9 (Gaussian skips no letter after I; libint's read_g94_basis_library
# follows the same convention). SP is the one two-shell label.
_L = {sym: [l] for l, sym in enumerate("SPDFGHIJKL")}
_L["SP"] = [0, 1]
_ELEMENT = re.compile(r"^([A-Z][a-z]?)\s+0\s*$")
_SHELL = re.compile(r"^([A-Z]{1,2})\s+(\d+)\s+[-+0-9.DdEe]+\s*$")


def _float(tok: str) -> float:
    return float(tok.replace("D", "E").replace("d", "e"))


@lru_cache(maxsize=None)
def _parse_g94(path: str) -> dict[int, list[tuple[int, tuple[float, ...]]]]:
    """Per-element ``(l, exponents)`` shells, in file order.

    An independent reading of the Gaussian94 format (not libint's parser), so
    the comparison is against the file and not against libint's view of it.
    SP shells expand to an s then a p shell, as libint stores them.
    """
    shells: dict[int, list[tuple[int, tuple[float, ...]]]] = {}
    lines = Path(path).read_text().splitlines()
    z = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("!"):
            continue
        if line == "****":
            z = None
            continue
        m = _ELEMENT.match(line)
        if m and z is None:
            z = _SYMBOL_TO_Z[m.group(1).lower()]
            shells.setdefault(z, [])
            continue
        m = _SHELL.match(line)
        if m and z is not None:
            nprim = int(m.group(2))
            exps = tuple(_float(lines[i + k].split()[0]) for k in range(nprim))
            i += nprim
            for l in _L[m.group(1)]:
                shells[z].append((l, exps))
            continue
        raise AssertionError(f"{path}: unparsed line {line!r}")
    return shells


def _all_element_molecule(zs):
    atoms = [vq.Atom(z, [0.0, 0.0, 8.0 * k]) for k, z in enumerate(zs)]
    return vq.Molecule(atoms, 0, 1 if sum(zs) % 2 == 0 else 2)


@lru_cache(maxsize=None)
def _loaded(name: str, zs: tuple[int, ...]):
    basis = vq.BasisSet(_all_element_molecule(list(zs)), name)
    per_atom: dict[int, list[tuple[int, tuple[float, ...]]]] = {
        k: [] for k in range(len(zs))
    }
    for s in basis.shells():
        per_atom[s.atom_index].append((s.l, tuple(s.exponents)))
    return {zs[k]: v for k, v in per_atom.items()}


def _assert_loads_own_file(name: str) -> None:
    path = basis_file_path(name)
    assert path is not None, f"{name}: no shipped .g94 file"
    assert path.parent == _library_basis_dir(), (
        f"{name}: registry resolves {path}, libint reads {_library_basis_dir()}"
    )
    expected = _parse_g94(str(path))
    assert expected, f"{name}: parsed no elements from {path}"
    zs = tuple(sorted(expected))
    loaded = _loaded(name, zs)
    for z in zs:
        want, got = expected[z], loaded[z]
        assert len(got) == len(want), (
            f"{name} Z={z}: BasisSet loaded {len(got)} shells, "
            f"{path.name} has {len(want)}"
        )
        for k, (w, g) in enumerate(zip(want, got)):
            assert g[0] == w[0], f"{name} Z={z} shell {k}: l {g[0]} != file {w[0]}"
            assert g[1] == pytest.approx(w[1], rel=1e-12, abs=0.0), (
                f"{name} Z={z} shell {k}: exponents differ from {path.name}"
            )


@pytest.mark.parametrize("name", PP_ORBITAL)
def test_pp_orbital_set_loads_its_shipped_file(name):
    _assert_loads_own_file(name)


@pytest.mark.parametrize("name", PP_RIFIT)
def test_pp_rifit_set_loads_its_shipped_file(name):
    _assert_loads_own_file(name)


@pytest.mark.parametrize("name", AE_AFFECTED)
def test_all_electron_aug_set_loads_its_shipped_file(name):
    _assert_loads_own_file(name)


def test_every_shipped_aug_file_is_covered():
    """A future ``aug-cc-pv*`` file cannot slip past the lists above."""
    shipped = set(_shipped_aug_names())
    explicit = set(PP_ORBITAL) | set(PP_RIFIT) | set(AE_AFFECTED)
    composed = {
        n for n in shipped
        if (_library_basis_dir() / f"augmentation-{n[4:]}.g94").is_file()
    }
    assert shipped - explicit - composed == set()


@pytest.mark.parametrize(
    "aug, base, symbol",
    [(f"aug-cc-pv{z}z-pp", f"cc-pv{z}z-pp", sym) for z in _ZETAS
     for sym in ("Br", "I", "Au", "Rn")]
    + [(f"aug-cc-pv({z}+d)z", f"cc-pv({z}+d)z", "Cl") for z in _ZETAS]
    + [(f"aug-cc-pv{z}z-rifit", f"cc-pv{z}z-rifit", "O") for z in _ZETAS],
)
def test_aug_set_is_larger_than_its_parent(aug, base, symbol):
    """The diffuse shells are present: aug has more shells and a smaller exponent."""
    z = _SYMBOL_TO_Z[symbol.lower()]
    mol = _all_element_molecule([z])
    a = vq.BasisSet(mol, aug)
    b = vq.BasisSet(mol, base)
    assert len(a.shells()) > len(b.shells()), (
        f"{aug} on {symbol} loaded {len(a.shells())} shells, "
        f"no more than {base} ({len(b.shells())})"
    )
    assert min(min(s.exponents) for s in a.shells()) < min(
        min(s.exponents) for s in b.shells()
    )


def test_composed_aug_sets_still_compose():
    """aug-cc-pVDZ ships only as cc-pVDZ + augmentation-cc-pVDZ; keep that path."""
    mol = _all_element_molecule([8])
    assert vq.BasisSet(mol, "aug-cc-pvdz").nbasis == 23
    assert vq.BasisSet(mol, "cc-pvdz").nbasis == 14


def _sad_density(mol, basis, n_occ):
    return vq._vibeqc_core._guess_closed_shell_density(
        mol, basis, n_occ, vq.InitialGuess.SAD, is_periodic=False
    )


def test_sad_guess_uses_the_same_aug_file():
    """The SAD atomic solve loads the basis by name too; it must see the same shells.

    A loader fix confined to BasisSet would give SAD the parent set's
    functions, and the per-atom block assembly would refuse the size
    mismatch.
    """
    mol = vq.Molecule(
        [vq.Atom(17, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 2.41])], 0, 1
    )
    basis = vq.BasisSet(mol, "aug-cc-pv(d+d)z")
    assert basis.nbasis > vq.BasisSet(mol, "cc-pv(d+d)z").nbasis
    D = _sad_density(mol, basis, 9)
    S = vq.compute_overlap(basis)
    assert D.shape == (basis.nbasis, basis.nbasis)
    assert float((D * S).sum()) == pytest.approx(18.0, abs=1e-8)
