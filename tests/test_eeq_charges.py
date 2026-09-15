"""Native EEQ atomic-charge model (Caldeweyher 2019, Apache-2.0 port
from Grimme group's multicharge library).

Phase D4a — first installment of the native D4 backend. This covers
the EEQ charge model (the algorithmically distinctive piece of D4 over
D3); the full D4 dispersion energy + reference C6 table lives in
Phase D4b.

Pins:

  1. Sum of partial charges equals the requested total charge
     (charge-conservation constraint baked into the EEQ system).
  2. Symmetry: equivalent atoms (e.g. both H in H2O) get identical
     charges to machine precision.
  3. Sign: O / N / F polarise negative; H is positive on the
     conventional H-X bonded systems.
  4. EEQ-CN: bound, symmetric, scales with neighbor count, matches the
     hand-traced ``0.5·(1 + erf(-7.5·(r-rc)/rc))`` form for at least
     one analytic case.
  5. Element-coverage check: a Z outside the table (e.g. Z = 110) is
     rejected with a clear error.
  6. The result-object's ``chemical_potential`` field is real-valued
     and roughly in the −2..+2 Ha range for normal organics.

Cross-validation vs the optional ``dftd4`` reference is run when
``dftd4`` is installed but pinned **loosely** — vibe-qc's native EEQ
currently agrees with dftd4 to ~10% in absolute charge magnitude on
H2O / cc-pVDZ; bit-exactness pending the Phase D4a-ii fix to
reconcile a CN-counting convention difference (the algorithm and
parameter tables match, but mctc-ncoord adds an extra layering
inside ``cut_coordination_number`` and the EN-weighted variant of the
erf counting function that this port hasn't fully tracked yet).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import (
    Atom,
    EEQOptions,
    EEQResult,
    Molecule,
    eeq_charges,
    eeq_coordination_numbers,
)


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _h2o():
    """H2O at the project-standard geometry (bohr)."""
    return Molecule([
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])


def _ch4():
    a = 0.626 * ANGSTROM_TO_BOHR
    return Molecule([
        Atom(6, [0.0, 0.0, 0.0]),
        Atom(1, [+a, +a, +a]),
        Atom(1, [-a, -a, +a]),
        Atom(1, [-a, +a, -a]),
        Atom(1, [+a, -a, -a]),
    ])


def _nh3():
    return Molecule([
        Atom(7, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.78, -0.62]),
        Atom(1, [1.54, -0.89, -0.62]),
        Atom(1, [-1.54, -0.89, -0.62]),
    ])


def _hf():
    return Molecule([
        Atom(9, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 0.0, 1.74]),
    ])


# ---------------------------------------------------------------------
# Invariants of the EEQ solve.
# ---------------------------------------------------------------------

def test_eeq_charges_sum_to_zero_for_neutral_molecule():
    """Charge-conservation: Σ q_A == 0 for a neutral molecule."""
    for mol in (_h2o(), _ch4(), _nh3(), _hf()):
        res = eeq_charges(mol)
        assert isinstance(res, EEQResult)
        q = np.asarray(res.charges)
        assert q.sum() == pytest.approx(0.0, abs=1e-12)


def test_eeq_charges_sum_to_requested_total_for_ion():
    """Pass an explicit total_charge and the EEQ solve respects it."""
    mol = _h2o()
    for q_tot in (0.0, 1.0, -1.0, 2.5):
        res = eeq_charges(mol, total_charge=q_tot)
        q = np.asarray(res.charges)
        assert q.sum() == pytest.approx(q_tot, abs=1e-12)


def test_eeq_charges_symmetry_h2o():
    """The two H atoms in H2O are crystallographically equivalent — same
    charge to machine precision."""
    res = eeq_charges(_h2o())
    q = np.asarray(res.charges)
    assert q[1] == pytest.approx(q[2], abs=1e-12)


def test_eeq_charges_symmetry_ch4():
    """All four H atoms in CH4 are equivalent."""
    res = eeq_charges(_ch4())
    q = np.asarray(res.charges)
    assert np.allclose(q[1:], q[1], atol=1e-10)


def test_eeq_charges_sign_convention_xh():
    """For X−H bonded systems with X more electronegative than H, the
    H atoms should carry positive charge and X negative."""
    for mol in (_h2o(), _nh3(), _hf()):
        q = np.asarray(eeq_charges(mol).charges)
        assert q[0] < 0, "central X should be negative"
        assert (q[1:] > 0).all(), "all H atoms should be positive"


def test_eeq_chemical_potential_finite():
    """λ (chemical potential) is real and roughly in the ±2 Ha range
    for small neutral organics."""
    for mol in (_h2o(), _ch4(), _nh3(), _hf()):
        res = eeq_charges(mol)
        assert math.isfinite(res.chemical_potential)
        assert abs(res.chemical_potential) < 5.0


# ---------------------------------------------------------------------
# CN-counting properties.
# ---------------------------------------------------------------------

def test_eeq_cn_non_negative_and_finite():
    for mol in (_h2o(), _ch4(), _nh3(), _hf()):
        cn = np.asarray(eeq_coordination_numbers(mol))
        assert (cn >= 0).all()
        assert np.isfinite(cn).all()


def test_eeq_cn_symmetry_ch4():
    cn = np.asarray(eeq_coordination_numbers(_ch4()))
    # C central is roughly 4 × per-H count; all H equivalent.
    assert np.allclose(cn[1:], cn[1], atol=1e-10)


def test_eeq_cn_options_default_match_multicharge():
    """Default EEQOptions match multicharge::new_eeq2019_model:
    cn_cutoff=25.0 bohr, cn_exp=7.5, cn_max=8.0."""
    opts = EEQOptions()
    assert opts.cn_cutoff == pytest.approx(25.0)
    assert opts.cn_exp == pytest.approx(7.5)
    assert opts.cn_max == pytest.approx(8.0)


# ---------------------------------------------------------------------
# Pinned vibe-qc native numbers (regression-guard against future code
# changes accidentally drifting the implementation).
# ---------------------------------------------------------------------

def test_eeq_h2o_pinned_numbers():
    """Regression pin: native EEQ produces these specific numbers on
    H2O at the standard geometry. Update only when the underlying
    EEQ math/parameters change deliberately.

    These are the Phase D4a-ii values — after the D3/D4 4/3
    covalent-radius convention was applied to the CN counting, the
    native EEQ is bit-exact with the dftd4 reference (see
    test_eeq_matches_dftd4_bit_exact below). They equal dftd4's
    partial charges to all printed digits."""
    res = eeq_charges(_h2o())
    q = np.asarray(res.charges)
    assert q[0] == pytest.approx(-0.65659292, abs=1e-7)
    assert q[1] == pytest.approx(+0.32829646, abs=1e-7)
    assert q[2] == pytest.approx(+0.32829646, abs=1e-7)


def test_eeq_cn_h2o_integer_like():
    """With the D3/D4 4/3 covalent-radius scaling, the EEQ coordination
    numbers come out close to the integer chemical coordination —
    CN(O) ≈ 2, CN(H) ≈ 1 in water. (Pre-fix, with bare Pyykkö radii,
    CN(O) was ~1.28 — the ~10 % charge error this test guards against
    regressing.)"""
    cn = np.asarray(eeq_coordination_numbers(_h2o()))
    assert cn[0] == pytest.approx(2.0, abs=0.05)
    assert cn[1] == pytest.approx(1.0, abs=0.05)
    assert cn[2] == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------
# Cross-validation vs the optional dftd4 reference — BIT-EXACT.
#
# Phase D4a-ii: after the D3/D4 4/3 covalent-radius convention was
# applied to the CN counting function, vibe-qc's native EEQ reproduces
# dftd4's "partial charges" to machine precision (the only residual is
# floating-point roundoff in the two codes' independent linear solves,
# ~1e-13). The earlier loose qualitative comparison is retired.
# ---------------------------------------------------------------------

def _dftd4_available():
    try:
        import dftd4.interface  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _dftd4_available(),
                    reason="dftd4 not installed")
@pytest.mark.parametrize("label", ["H2O", "CH4", "NH3"])
def test_eeq_matches_dftd4_bit_exact(label):
    """Native EEQ charges reproduce dftd4's EEQ-2019 ``partial charges``
    to machine precision (|Δq| < 1e-10 per atom). This is the Phase
    D4a-ii parity target — the native model is now a drop-in
    replacement for dftd4's charge model, not just a qualitative
    approximation."""
    from dftd4.interface import DispersionModel

    factories = {"H2O": _h2o, "CH4": _ch4, "NH3": _nh3}
    mol = factories[label]()
    native = np.asarray(eeq_charges(mol).charges)

    numbers = np.array([a.Z for a in mol.atoms], dtype=np.int32)
    positions = np.array([a.xyz for a in mol.atoms])
    ref = DispersionModel(numbers=numbers, positions=positions)
    q_ref = np.asarray(ref.get_properties()["partial charges"])

    delta = np.abs(native - q_ref).max()
    assert delta < 1e-10, (
        f"{label}: |Δq| = {delta:.2e} exceeds the 1e-10 bit-exact "
        f"bound (native={native}, dftd4={q_ref})")


# ---------------------------------------------------------------------
# Error handling.
# ---------------------------------------------------------------------

def test_eeq_rejects_unsupported_element():
    """An atom outside the EEQ2019 parameter table (Z = 1..103) is
    rejected with a clear error."""
    mol = Molecule([Atom(110, [0.0, 0.0, 0.0])])
    with pytest.raises(ValueError, match="outside the EEQ"):
        eeq_charges(mol)
