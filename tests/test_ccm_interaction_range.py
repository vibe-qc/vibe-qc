"""Interaction-range parameterization of the AICCM cluster (Task A).

The cyclic cluster can be sized by a **real-space interaction radius** ``R_c``
instead of an explicit mesh ``nrep`` — the geometric real-space dual of a
finite translation-group character mesh. This does not identify Γ-CCM with
χ-CCM or with a neutral GDF control. The minimal cluster whose supercell Wigner–Seitz cell
encloses a sphere of radius ``R_c`` around every atom is derived automatically.

These tests prove, against an independent brute-force oracle, the geometric
guarantee the derivation rests on — the WS inscribed-sphere radius
``r_in = λ₁/2`` of the derived cluster is ``≥ R_c`` for *every* lattice — and
that the radius path is numerically identical to the explicit-``nrep`` path.

Derivation: docs/aiccm2026dev_a_followon.md § "Interaction-range
parameterization". Theory: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import (
    CCMSystem,
    ccm_interaction_range_scan,
    interplanar_spacings,
    kspacing_for_interaction_range,
    nrep_for_interaction_range,
    wsc_inscribed_radius,
)
from vibeqc.periodic.ccm.scf import run_ccm_rhf
from vibeqc.periodic.ccm.wigner_seitz import shortest_lattice_vector_length

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

BOHR = 1.0 / 0.529177210903


# --------------------------------------------------------------------------- #
# A battery of lattices spanning the relevant cell shapes.
# --------------------------------------------------------------------------- #
LATTICES = {
    "cubic": np.diag([4.0, 4.0, 4.0]),
    "ortho": np.diag([3.0, 5.0, 7.0]),
    "tetragonal": np.diag([3.0, 3.0, 6.0]),
    "fcc_primitive": 4.0 * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]),
    "bcc_primitive": 4.0 * np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]),
    "hexagonal": np.array([[4.0, 0, 0], [-2.0, 2.0 * np.sqrt(3.0), 0], [0, 0, 6.0]]),
    "triclinic": np.array([[4.0, 0, 0], [1.5, 4.2, 0], [0.8, 1.1, 5.0]]),
    "vacuum_chain": np.diag([7.5589, 40.0, 40.0]),
}


def _brute_shortest(L, shell=6):
    """Independent oracle: shortest nonzero lattice vector by enumeration."""
    rng = range(-shell, shell + 1)
    best = np.inf
    for i in rng:
        for j in rng:
            for k in rng:
                if i == j == k == 0:
                    continue
                best = min(best, float(np.linalg.norm(i * L[0] + j * L[1] + k * L[2])))
    return best


# --------------------------------------------------------------------------- #
# interplanar_spacings
# --------------------------------------------------------------------------- #
def test_interplanar_spacings_orthogonal():
    d = interplanar_spacings(np.diag([3.0, 5.0, 7.0]))
    assert d == pytest.approx([3.0, 5.0, 7.0])


def test_interplanar_spacings_vacuum():
    d = interplanar_spacings(np.diag([7.5589, 40.0, 40.0]))
    assert d == pytest.approx([7.5589, 40.0, 40.0])


def test_interplanar_spacings_singular_raises():
    with pytest.raises(ValueError):
        interplanar_spacings(np.array([[1.0, 0, 0], [2.0, 0, 0], [0, 0, 1.0]]))


# --------------------------------------------------------------------------- #
# shortest lattice vector / inscribed radius vs brute-force oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(LATTICES))
def test_shortest_vector_matches_oracle(name):
    L = LATTICES[name]
    lam = shortest_lattice_vector_length(L)
    assert lam == pytest.approx(_brute_shortest(L), abs=1e-9)


@pytest.mark.parametrize("name", list(LATTICES))
def test_inscribed_radius_is_half_shortest(name):
    L = LATTICES[name]
    assert wsc_inscribed_radius(L) == pytest.approx(0.5 * _brute_shortest(L), abs=1e-9)


# --------------------------------------------------------------------------- #
# THE central guarantee: the derived cluster's WS cell encloses the R_c sphere.
# r_in(L_c) >= R_c  AND  the per-direction bound  min_i N_i d_i >= 2 R_c.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(LATTICES))
@pytest.mark.parametrize("Rc", [2.5, 5.0, 8.0, 13.0, 20.0])
def test_inscribed_sphere_gate(name, Rc):
    L = LATTICES[name]
    d = interplanar_spacings(L)
    nrep = nrep_for_interaction_range(L, Rc, dim=3)
    Lc = np.asarray(nrep, dtype=float)[:, None] * L
    # 1) the rigorous inscribed-sphere condition (the real guarantee)
    assert wsc_inscribed_radius(Lc) >= Rc - 1e-9
    # 2) the per-direction sufficient bound used to derive nrep
    assert (np.asarray(nrep) * d).min() >= 2.0 * Rc - 1e-9


def test_orthogonal_bound_is_tight():
    """For an orthogonal cell N_i = ceil(2 R_c / a_i) exactly (bound is tight)."""
    L = np.diag([3.0, 5.0, 7.0])
    Rc = 7.0
    nrep = nrep_for_interaction_range(L, Rc, dim=3)
    assert nrep == (int(np.ceil(14.0 / 3.0)), int(np.ceil(14.0 / 5.0)), int(np.ceil(14.0 / 7.0)))
    assert nrep == (5, 3, 2)


def test_nrep_monotone_in_radius():
    L = LATTICES["triclinic"]
    prev = (0, 0, 0)
    for Rc in [1.0, 2.0, 4.0, 8.0, 16.0]:
        nrep = nrep_for_interaction_range(L, Rc, dim=3)
        assert all(n >= p for n, p in zip(nrep, prev))  # componentwise non-decreasing
        prev = nrep


def test_dim_caps_nonperiodic_directions():
    L = np.diag([4.0, 4.0, 4.0])
    # dim=1 → only the first direction extends
    assert nrep_for_interaction_range(L, 10.0, dim=1) == (5, 1, 1)
    assert nrep_for_interaction_range(L, 10.0, dim=2) == (5, 5, 1)
    assert nrep_for_interaction_range(L, 10.0, dim=3) == (5, 5, 5)


def test_vacuum_self_limits():
    """Vacuum-padded directions stay at N=1 without a dim cap (large d)."""
    L = np.diag([6.0, 40.0, 40.0])
    assert nrep_for_interaction_range(L, 12.0, dim=3) == (4, 1, 1)


def test_vacuum_breakdown_threshold():
    """A vacuum direction self-limits only up to R_c = half the padding.

    With 40-bohr vacuum the WS half-width along y/z is 20 bohr; a radius beyond
    that legitimately forces the vacuum direction to grow (the sphere no longer
    fits), so dim=3 reports it rather than silently truncating.
    """
    L = np.diag([6.0, 40.0, 40.0])
    assert nrep_for_interaction_range(L, 19.0, dim=3) == (7, 1, 1)   # ⌈38/6⌉=7, y/z still 1
    assert nrep_for_interaction_range(L, 24.0, dim=3) == (8, 2, 2)   # 24>20 → y,z grow
    # an explicit dim cap keeps a genuine low-D system 1-D regardless of R_c
    assert nrep_for_interaction_range(L, 24.0, dim=1) == (8, 1, 1)


def test_radius_must_be_positive():
    with pytest.raises(ValueError):
        nrep_for_interaction_range(np.diag([4.0, 4.0, 4.0]), -1.0)


def test_kspacing_duality():
    assert kspacing_for_interaction_range(10.0) == pytest.approx(np.pi / 10.0)
    # larger radius ⇔ denser mesh (smaller Δk)
    assert kspacing_for_interaction_range(20.0) < kspacing_for_interaction_range(10.0)


# --------------------------------------------------------------------------- #
# CCMSystem API
# --------------------------------------------------------------------------- #
def _h2_chain(cell=6.0):
    return PeriodicSystem(
        3, np.diag([cell, 40.0, 40.0]),
        [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], charge=0, multiplicity=1,
    )


def test_from_interaction_range_derives_expected_nrep():
    unit = _h2_chain(6.0)
    ccm = CCMSystem.from_interaction_range(unit, 12.0, "sto-3g")
    assert ccm.nrep == (4, 1, 1)
    assert ccm.interaction_range == pytest.approx(12.0)
    assert ccm.wsc_inscribed_radius >= 12.0 - 1e-9


def test_angstrom_convenience_matches_bohr():
    unit = _h2_chain(6.0)
    rc_bohr = 12.0
    a = CCMSystem(unit, basis="sto-3g", interaction_range=rc_bohr)
    b = CCMSystem(unit, basis="sto-3g", interaction_range_ang=rc_bohr * 0.529177210903)
    c = CCMSystem.from_interaction_range(unit, rc_bohr * 0.529177210903, "sto-3g", units="angstrom")
    assert a.nrep == b.nrep == c.nrep


def test_radius_path_equals_explicit_nrep():
    """Radius- and explicit-nrep clusters are bit-identical (overlap + energy)."""
    unit = _h2_chain(6.0)
    ccm_r = CCMSystem.from_interaction_range(unit, 12.0, "sto-3g")
    ccm_n = CCMSystem(unit, (4, 1, 1), "sto-3g")
    assert ccm_r.nrep == ccm_n.nrep
    from vibeqc.periodic.ccm import ccm_overlap
    assert np.allclose(ccm_overlap(ccm_r), ccm_overlap(ccm_n), atol=1e-14)
    er = run_ccm_rhf(ccm_r, method="aiccm2026dev-a").energy_per_atom
    en = run_ccm_rhf(ccm_n, method="aiccm2026dev-a").energy_per_atom
    assert er == pytest.approx(en, abs=1e-12)


def test_explicit_nrep_has_no_requested_radius():
    ccm = CCMSystem(_h2_chain(6.0), (3, 1, 1), "sto-3g")
    assert ccm.interaction_range is None
    assert ccm.wsc_inscribed_radius == pytest.approx(0.5 * 3 * 6.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(nrep=(2, 1, 1), interaction_range=10.0),     # both
        dict(),                                            # neither
        dict(interaction_range=1.0, interaction_range_ang=1.0),  # both units
    ],
)
def test_ambiguous_size_specs_raise(kwargs):
    unit = _h2_chain(6.0)
    with pytest.raises(ValueError):
        CCMSystem(unit, basis="sto-3g", **kwargs)


def test_missing_basis_raises():
    with pytest.raises(ValueError):
        CCMSystem(_h2_chain(6.0), (2, 1, 1))


def test_bad_units_raise():
    with pytest.raises(ValueError):
        CCMSystem.from_interaction_range(_h2_chain(6.0), 10.0, "sto-3g", units="nm")


# --------------------------------------------------------------------------- #
# Explicit nrep validation -- no silent truncation (issue #390)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "nrep",
    [
        (1.9, 1, 1),          # float component, was truncated to 1
        (1.0, 1, 1),          # integer-valued float, was coerced silently
        (True, 1, 1),         # bool is an int subclass, was accepted as 1
        (False, 1, 1),
        ("2", 1, 1),         # string numeral, was int()-coerced
        "211",                # whole-string, iterable of chars
        (np.nan, 1, 1),       # non-finite component
        (np.inf, 1, 1),
        np.array([2.7, 1, 1]),  # NumPy float array, was truncated
        (2, 1),               # wrong length
        (2, 1, 1, 1),
        2,                    # not a sequence at all
        np.array([[2], [1], [1]]),  # nested components
    ],
    ids=[
        "float-component", "integer-valued-float", "bool-true", "bool-false",
        "string-component", "whole-string", "nan", "inf", "np-float-array",
        "short", "long", "scalar", "nested-array",
    ],
)
def test_explicit_nrep_rejects_non_integer_components(nrep, monkeypatch):
    """Issue #390: an explicit mesh is not coerced; it is exact or refused.

    ``int(x)`` conversion silently rewrote (1.9, 1, 1) to (1, 1, 1) and ran a
    different cyclic Hamiltonian than the caller requested. Every non-integer
    or boolean component must raise a bounded ValueError before any supercell
    or basis work (pinned here by asserting _build_supercell never runs).
    """
    unit = _h2_chain(6.0)
    calls: list[object] = []
    monkeypatch.setattr(
        CCMSystem, "_build_supercell", lambda self, us: calls.append(us)
    )
    with pytest.raises(ValueError):
        CCMSystem(unit, nrep, "sto-3g")
    assert calls == []


@pytest.mark.parametrize(
    "nrep,expected",
    [
        ((3, 1, 1), (3, 1, 1)),                     # Python ints
        (np.array([3, 1, 1]), (3, 1, 1)),           # NumPy int64 array
        (np.array([3, 1, 1], dtype=np.int32), (3, 1, 1)),
        ([np.int64(3), 1, 1], (3, 1, 1)),           # mixed int kinds
        ((np.int32(3), np.int64(1), 1), (3, 1, 1)),
    ],
    ids=["python-ints", "np-int64-array", "np-int32-array",
         "mixed-list", "mixed-scalars"],
)
def test_explicit_nrep_accepts_integer_sequences(nrep, expected):
    """Exactly three Python/NumPy integer components stay accepted and are
    stored without coercion (issue #390 closure criteria)."""
    ccm = CCMSystem(_h2_chain(6.0), nrep, "sto-3g")
    assert ccm.nrep == expected
    assert all(isinstance(n, int) for n in ccm.nrep)


def test_explicit_nrep_zero_or_negative_component_raises():
    unit = _h2_chain(6.0)
    for nrep in [(0, 1, 1), (-2, 1, 1), (2, 1, 0)]:
        with pytest.raises(ValueError):
            CCMSystem(unit, nrep, "sto-3g")


# --------------------------------------------------------------------------- #
# Convergence scan (the real-space dual of a k-mesh convergence study)
# --------------------------------------------------------------------------- #
def test_interaction_range_scan_converges():
    unit = _h2_chain(6.0)
    # keep R_c below the 20-bohr vacuum half-width so the chain stays 1-D
    scan = ccm_interaction_range_scan(
        unit, "sto-3g", radii_bohr=[6.0, 12.0, 18.0], method="aiccm2026dev-a"
    )
    # one distinct cluster per radius here (nrep strictly grows along the chain)
    assert [p.nrep for p in scan.points] == [(2, 1, 1), (4, 1, 1), (6, 1, 1)]
    # the k-mesh dual equals the cluster mesh
    assert scan.points[1].kmesh_equivalent == (4, 1, 1)
    # converges: successive |ΔE/atom| shrinks toward zero
    d = np.abs(scan.deltas[1:])
    assert d[-1] < d[0]
    assert d[-1] < 1e-3
    # converged_radius returns a real radius at a loose tol, None at an impossible one
    assert scan.converged_radius(1e-3) is not None
    assert scan.converged_radius(0.0) is None
    # records are JSON-friendly
    rec = scan.as_records()[0]
    assert set(rec) >= {"radius_bohr", "nrep", "kmesh_equivalent", "energy_per_atom"}


def test_scan_skips_duplicate_clusters():
    unit = _h2_chain(6.0)
    # on a 6-bohr cell N=⌈R_c/3⌉, so 10/11/12 all map to (4,1,1); duplicates skipped
    scan = ccm_interaction_range_scan(
        unit, "sto-3g", radii_bohr=[10.0, 11.0, 12.0, 18.0], method="aiccm2026dev-a"
    )
    assert [p.nrep for p in scan.points] == [(4, 1, 1), (6, 1, 1)]
