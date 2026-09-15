"""CODATA atomic enthalpies of formation at 0 K.

Used to derive a molecule's atomization energy from its
``Hfg(0K)`` value (which CCCBDB's ``exp2x`` per-molecule page does
tabulate) -- far cleaner than scraping CCCBDB's ``atomize1x``
comparison page, which turned out to be form-only and doesn't
honour ``?casno=``.

Atomization energy of a molecule M_n with composition ``S aᵢ Eᵢ``:

    AE(M, 0K) = S aᵢ . Hf°(Eᵢ_atom, 0K) - Hf°(M, 0K)

(positive number when the molecule is bound -- the energy that has
to be supplied to break it into free gas-phase atoms).

Values below are from CODATA 2018 / JANAF tables, published per
CCCBDB's source documentation (cccbdb.nist.gov/hf0k.asp); kJ/mol.
We restrict to elements present in the v1 canonical molecule set
(H, C, N, O, F, P, S, Cl, Br + a few extras likely to come up
soon). Add more on demand -- but cite the source on every addition.

Uncertainty values are CODATA 2018's published 1-s uncertainties.
"""
from __future__ import annotations

from typing import NamedTuple


class AtomicEnthalpy(NamedTuple):
    """One element's atomic enthalpy of formation at 0 K."""
    symbol: str
    z: int
    h_f_0k_kj_per_mol: float
    uncertainty_kj_per_mol: float
    source: str


# CODATA 2018 / JANAF 4th edition atomic Hf°(0K) values, kJ/mol.
# Cross-checked against CCCBDB cccbdb.nist.gov/hf0k.asp (Aug 2024 retrieval).
ATOMIC_HF_0K_KJ_PER_MOL: dict[str, AtomicEnthalpy] = {
    "H":  AtomicEnthalpy("H",  1, 216.034, 0.000, "CODATA 2018"),
    "Li": AtomicEnthalpy("Li", 3, 157.71,  0.13,  "CODATA 2018"),
    "Be": AtomicEnthalpy("Be", 4, 320.45,  4.18,  "JANAF"),
    "B":  AtomicEnthalpy("B",  5, 565.,    5.,    "JANAF"),
    "C":  AtomicEnthalpy("C",  6, 711.19,  0.46,  "CODATA 2018 (graphite ref)"),
    "N":  AtomicEnthalpy("N",  7, 470.59,  0.10,  "CODATA 2018 (1/2 N2 ref)"),
    "O":  AtomicEnthalpy("O",  8, 246.79,  0.10,  "CODATA 2018 (1/2 O2 ref)"),
    "F":  AtomicEnthalpy("F",  9,  77.27,  0.30,  "CODATA 2018 (1/2 F2 ref)"),
    "Na": AtomicEnthalpy("Na", 11, 107.50, 0.70,  "JANAF"),
    "Mg": AtomicEnthalpy("Mg", 12, 145.90, 0.80,  "JANAF"),
    "Al": AtomicEnthalpy("Al", 13, 327.32, 1.30,  "JANAF"),
    "Si": AtomicEnthalpy("Si", 14, 446.0,  8.,    "JANAF"),
    "P":  AtomicEnthalpy("P",  15, 316.5,  1.0,   "JANAF (white-P ref)"),
    "S":  AtomicEnthalpy("S",  16, 274.92, 0.20,  "CODATA 2018 (S(rhomb) ref)"),
    "Cl": AtomicEnthalpy("Cl", 17, 119.621, 0.008, "CODATA 2018 (1/2 Cl2 ref)"),
    "K":  AtomicEnthalpy("K",  19,  89.0,  0.8,   "JANAF"),
    "Ca": AtomicEnthalpy("Ca", 20, 177.8,  0.8,   "JANAF"),
    "Br": AtomicEnthalpy("Br", 35, 117.93, 0.02,  "CODATA 2018 (1/2 Br2(g) ref)"),
}


def atomic_h_f_0k(symbol: str) -> float:
    """Atomic Hf°(0K) in kJ/mol for ``symbol``.

    Raises ``KeyError`` for unsupported elements -- the caller is
    expected to handle this by skipping the atomization-energy
    derivation for that molecule (the rest of the
    ``ExperimentalReference`` is still well-defined).
    """
    if symbol not in ATOMIC_HF_0K_KJ_PER_MOL:
        raise KeyError(
            f"no atomic Hf°(0K) for element {symbol!r}; extend "
            f"references/atomic_enthalpies.py with a CODATA / JANAF value."
        )
    return ATOMIC_HF_0K_KJ_PER_MOL[symbol].h_f_0k_kj_per_mol


def derive_atomization_kj_per_mol(
    *,
    composition: dict[str, int],
    h_f_0k_molecule_kj_per_mol: float,
) -> float:
    """Derive AE(M, 0K) in kJ/mol from molecular Hf°(0K) + atomic refs.

    ``composition``: mapping of element symbol -> count, e.g. ``{"H": 2,
    "O": 1}`` for water.

    Sign convention: positive when the molecule is bound (energy
    required to dissociate into atoms).
    """
    sum_atomic = 0.0
    for sym, count in composition.items():
        sum_atomic += count * atomic_h_f_0k(sym)
    return sum_atomic - h_f_0k_molecule_kj_per_mol
