"""``vibeqc.fetch.heuristics`` — pure-function decision tables.

Pins the (initial-guess, damping) trigger set we copied from the
existing periodic regression specs. Drift here means a fetched SPEC
suddenly disagrees with the hand-curated SPEC for the same chemistry
— catch that statically rather than in an SCF that won't converge.
"""
from __future__ import annotations

import pytest

from vibeqc.fetch.heuristics import (
    molecular_multiplicity,
    open_shell_default,
    pick_damping,
    pick_initial_guess,
    pick_kmesh,
    pick_recommended_basis,
)


@pytest.mark.parametrize(
    "symbols,want_guess,want_damping",
    [
        # Ionic / deep-core → SAD + 0.85.
        (["Mg", "O"],   "SAD",   0.85),
        (["Na", "Cl"],  "SAD",   0.85),
        (["Li", "H"],   "SAD",   0.85),
        (["Al", "O"],   "SAD",   0.85),
        # Wide-gap / light covalent → HCORE + 0.5.
        (["H", "H"],    "HCORE", 0.5),
        (["C", "C"],    "HCORE", 0.5),
        (["Si", "Si"],  "HCORE", 0.5),
        (["Ne"],        "HCORE", 0.5),
        # F is intentionally NOT a SAD trigger — HF / NeF regression
        # specs run fine on HCORE+0.5 (kept consistent with the table
        # in heuristics.py).
        (["H", "F"],    "HCORE", 0.5),
    ],
)
def test_initial_guess_and_damping(symbols, want_guess, want_damping):
    assert pick_initial_guess(symbols) == want_guess
    assert pick_damping(symbols) == pytest.approx(want_damping)


@pytest.mark.parametrize(
    "n_atoms,expected",
    [
        (1, (4, 4, 4)),
        (4, (4, 4, 4)),
        (5, (2, 2, 2)),
        (20, (2, 2, 2)),
        (21, (1, 1, 1)),
        (200, (1, 1, 1)),
    ],
)
def test_kmesh_seed(n_atoms, expected):
    assert pick_kmesh(n_atoms) == expected


def test_recommended_basis_periodic():
    # § 6.3: every periodic row lands on pob-tzvp.
    assert pick_recommended_basis(
        symbols=["Mg", "O"], is_periodic=True, n_atoms=8,
    ) == "pob-tzvp"
    assert pick_recommended_basis(
        symbols=["Si"], is_periodic=True, n_atoms=2,
    ) == "pob-tzvp"


def test_recommended_basis_molecular():
    assert pick_recommended_basis(
        symbols=["H", "H"], is_periodic=False, n_atoms=2,
    ) == "def2-svp"
    assert pick_recommended_basis(
        symbols=["C"] * 6 + ["H"] * 6, is_periodic=False, n_atoms=12,
    ) == "def2-svp"


def test_recommended_basis_quick_overrides():
    assert pick_recommended_basis(
        symbols=["Mg", "O"], is_periodic=True, n_atoms=8, quick=True,
    ) == "sto-3g"
    assert pick_recommended_basis(
        symbols=["H"] * 2, is_periodic=False, n_atoms=2, quick=True,
    ) == "sto-3g"


def test_open_shell_default_periodic_diamagnetic():
    is_open, mag = open_shell_default(is_periodic=True)
    assert is_open is False and mag is None


def test_open_shell_default_periodic_magnetic():
    is_open, mag = open_shell_default(
        is_periodic=True,
        mp_is_magnetic=True,
        mp_total_magnetization=2.5,
    )
    assert is_open is True


def test_open_shell_default_periodic_below_threshold():
    # Tiny residual magnetisation — treat as diamagnetic. Wrong-side
    # default is a UKS guess for an insulator; better conservative.
    is_open, mag = open_shell_default(
        is_periodic=True,
        mp_is_magnetic=True,
        mp_total_magnetization=0.05,
    )
    assert is_open is False and mag is None


def test_molecular_multiplicity_parity():
    assert molecular_multiplicity(2) == 1   # H2
    assert molecular_multiplicity(8) == 1   # CH4
    assert molecular_multiplicity(7) == 2   # OH radical
    assert molecular_multiplicity(9) == 2   # NH4 (hypothetical)
