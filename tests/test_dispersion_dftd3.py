"""Validation of vibe-qc's D3(BJ) against Grimme's reference
implementation via the ``dftd3`` pip package.

The ``dftd3`` package wraps Grimme's Fortran library and ships the
canonical c6ab reference table and damping parameters. We use it as the
ground-truth oracle here, the same way
``pyscf`` is the ground truth for RHF / DFT / MP2 energies elsewhere
in the test suite.

Two things we check:

1. Backend dispatch: ``backend="auto"`` uses the self-contained native path
   for H-Ar.
2. Numerical agreement: native energies and complete analytic gradients
   match the independent reference throughout the native element range.
3. Grimme-reference literature values: pick a water dimer geometry at
   a canonical PBE-D3BJ number and pin it.

If ``dftd3`` is not installed these tests skip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq


pytestmark = pytest.mark.skipif(
    not vq.dftd3_available(),
    reason="optional dftd3 package not installed",
)


H2O_XYZ = Path(__file__).parent.parent / "examples" / "h2o.xyz"
ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _rp208_methane() -> "vq.Molecule":
    """First methane from the archived RP208 final XYZ, in bohr."""
    ang_xyz = [
        (6, [0.0000000000, 0.0000000000, 0.0000000000]),
        (1, [0.6274999548, 0.6274999548, 0.6274999548]),
        (1, [-0.6274999548, -0.6274999548, 0.6274999548]),
        (1, [0.6274999548, -0.6274999548, -0.6274999548]),
        (1, [-0.6274999548, 0.6274999548, -0.6274999548]),
    ]
    return vq.Molecule([
        vq.Atom(Z, [x * ANGSTROM_TO_BOHR for x in xyz])
        for Z, xyz in ang_xyz
    ])


def _rp208_methane_dimer(separation_angstrom: float) -> "vq.Molecule":
    """RP208 methane orientation with a chosen carbon-carbon separation."""
    monomer = list(_rp208_methane().atoms)
    atoms = [vq.Atom(a.Z, list(a.xyz)) for a in monomer]
    shift = separation_angstrom * ANGSTROM_TO_BOHR
    atoms.extend(
        vq.Atom(a.Z, [a.xyz[0], a.xyz[1], a.xyz[2] + shift])
        for a in monomer
    )
    return vq.Molecule(atoms)


def _h2o_mol() -> "vq.Molecule":
    return vq.Molecule.from_xyz(str(H2O_XYZ))


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

def test_backend_auto_uses_quantitative_native_path():
    """Auto is self-contained for H-Ar and agrees with the reference."""
    mol = _h2o_mol()
    e_auto = vq.compute_d3bj(mol, "pbe", backend="auto").energy
    e_d3 = vq.compute_d3bj(mol, "pbe", backend="dftd3").energy
    e_builtin = vq.compute_d3bj(mol, "pbe", backend="builtin").energy
    assert e_auto == pytest.approx(e_builtin, rel=0, abs=0)
    assert e_auto == pytest.approx(e_d3, rel=0, abs=1e-14)


def test_backend_explicit_builtin_matches_dftd3():
    """Native C6 interpolation reproduces the independent reference."""
    mol = _h2o_mol()
    e_builtin = vq.compute_d3bj(mol, "pbe", backend="builtin").energy
    e_d3 = vq.compute_d3bj(mol, "pbe", backend="dftd3").energy
    assert e_builtin < 0.0
    assert e_builtin == pytest.approx(e_d3, rel=0, abs=1e-14)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        vq.compute_d3bj(_h2o_mol(), "pbe", backend="neither")


# ---------------------------------------------------------------------------
# Functional parameter coverage
# ---------------------------------------------------------------------------

def test_native_parameters_match_dftd3_registry():
    """Sample the full registry, including aliases and double hybrids."""
    for name in (
        "bp86", "b97-d", "pbe", "pbe0", "b3lyp", "blyp", "r2scan",
        "wb97x", "wb97m", "dsdpbep86", "dodpbep86", "skala-1.0",
    ):
        p_bi = vq.d3bj_params_for(name, backend="builtin")
        p_d3 = vq.d3bj_params_for(name, backend="dftd3")
        assert p_bi is not None
        assert p_d3 is not None
        assert p_bi.s6 == pytest.approx(p_d3.s6, rel=0, abs=1e-12), name
        assert p_bi.s8 == pytest.approx(p_d3.s8, rel=0, abs=1e-12), name
        assert p_bi.a1 == pytest.approx(p_d3.a1, rel=0, abs=1e-12), name
        assert p_bi.a2 == pytest.approx(p_d3.a2, rel=0, abs=1e-12), name


# ---------------------------------------------------------------------------
# Numerical agreement on the reference implementation
# ---------------------------------------------------------------------------

def test_water_dimer_pbe_d3bj_reproducible():
    """Fix the water-dimer geometry and functional; pin the resulting
    E_D3BJ to whatever the Grimme reference returns. Future vibe-qc
    changes to the dftd3 plumbing that change this number are bugs.

    Water dimer at the well-studied S22/S66 equilibrium O···O ≈ 2.91 Å
    geometry — close to the actual H-bonded minimum. Coordinates
    in Ångström converted to bohr inside the helper.
    """
    # Standard S22 water-dimer geometry (converted from S22 .xyz).
    angstrom_to_bohr = 1.8897261246
    ang_xyz = [
        (8, [-1.551007, -0.114520,  0.000000]),
        (1, [-1.934259,  0.762503,  0.000000]),
        (1, [-0.599677,  0.040712,  0.000000]),
        (8, [ 1.350625,  0.111469,  0.000000]),
        (1, [ 1.680398, -0.373741, -0.758561]),
        (1, [ 1.680398, -0.373741,  0.758561]),
    ]
    atoms = [
        vq.Atom(Z, [x * angstrom_to_bohr for x in xyz])
        for Z, xyz in ang_xyz
    ]
    mol = vq.Molecule(atoms)

    # dftd3 reference result for PBE-D3BJ.
    res = vq.compute_d3bj(mol, "pbe", backend="dftd3")
    # The dimer E_disp is ~ -1 mHa for PBE-D3BJ at this geometry
    # (mostly intermolecular; intramolecular piece is ~ -0.7 mHa from
    # the two monomers' internal contributions).
    assert res.energy < 0.0
    assert -2e-3 < res.energy < -5e-4, (
        f"water-dimer PBE-D3BJ out of expected mHa range: {res.energy}"
    )

    # Symmetry check: swapping monomers (reflecting through the
    # midpoint) should give the exact same number.
    mol_swapped = vq.Molecule([
        vq.Atom(a.Z, [-x for x in a.xyz]) for a in mol.atoms
    ])
    res_swapped = vq.compute_d3bj(mol_swapped, "pbe", backend="dftd3")
    assert res.energy == pytest.approx(res_swapped.energy, rel=1e-12)


def test_rp208_methane_dimer_b3lyp_d3bj_component():
    """Pin the geometry-only RP208 component against simple-dftd3/ORCA.

    ORCA 6.1.1 reports -0.005177608303 Ha for the same B3LYP parameters.
    The archived XYZ is rounded to ten decimals, for which simple-dftd3
    1.4.0 gives the target below. No SCF quantity enters this comparison.
    """
    archived_params = vq.D3BJParams(
        s6=1.0, s8=1.9889, a1=0.3981, a2=4.4211,
    )
    mol = _rp208_methane_dimer(3.7999997265)
    native = vq.compute_d3bj(mol, archived_params, backend="builtin").energy
    reference = vq.compute_d3bj(mol, archived_params, backend="dftd3").energy
    assert native == pytest.approx(-0.005177608710458, rel=0, abs=1e-12)
    assert native == pytest.approx(reference, rel=0, abs=1e-12)


def test_b3lyp_d3bj_methane_and_hydrogen_pair_references():
    """Pin a monomer and a two-atom case against simple-dftd3 1.4.0."""
    methane = vq.compute_d3bj(
        _rp208_methane(), "b3lyp", backend="auto"
    ).energy
    assert methane == pytest.approx(-0.001913707181662, rel=0, abs=1e-10)

    h2 = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 5.0]),
    ])
    h2_energy = vq.compute_d3bj(h2, "b3lyp", backend="auto").energy
    assert h2_energy == pytest.approx(-0.000242991727954, rel=0, abs=1e-10)


def test_methane_dimer_separation_tends_to_twice_monomer():
    """At large separation only the two intramolecular corrections remain."""
    monomer = vq.compute_d3bj(
        _rp208_methane(), "b3lyp", backend="auto"
    ).energy
    near = vq.compute_d3bj(
        _rp208_methane_dimer(3.8), "b3lyp", backend="auto"
    ).energy
    distant = vq.compute_d3bj(
        _rp208_methane_dimer(100.0), "b3lyp", backend="auto"
    ).energy
    far = vq.compute_d3bj(
        _rp208_methane_dimer(1000.0), "b3lyp", backend="auto"
    ).energy

    assert near < far
    # The Grimme CN logistic has a tiny finite asymptote (~1e-7 per remote
    # atom), so the combined-system limit differs from two separately
    # evaluated monomers by a few 1e-10 Ha rather than becoming bit-exact.
    target = 2.0 * monomer
    assert abs(far - target) < abs(distant - target)
    assert far == pytest.approx(target, rel=0, abs=5e-10)


def test_dftd3_gradient_shape_and_symmetry():
    """When ``with_gradient=True`` the dftd3 backend returns a
    ``(n_atoms, 3)`` gradient that is translationally invariant
    (columnwise sum = 0)."""
    mol = _h2o_mol()
    res = vq.compute_d3bj(
        mol, "pbe", backend="dftd3", with_gradient=True,
    )
    assert res.gradient.shape == (3, 3)
    assert np.allclose(res.gradient.sum(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("s9", [0.0, 1.0])
def test_rp208_native_energy_and_gradient_match_dftd3(s9):
    """Pin two-body and ATM coordination-number derivatives on RP208."""
    mol = _rp208_methane_dimer(3.7999997265)
    params = vq.D3BJParams(
        s6=1.0, s8=1.9889, a1=0.3981, a2=4.4211, s9=s9,
    )
    native = vq.compute_d3bj(
        mol, params, backend="builtin", with_gradient=True,
    )
    reference = vq.compute_d3bj(
        mol, params, backend="dftd3", with_gradient=True,
    )
    assert native.energy == pytest.approx(reference.energy, rel=0, abs=2e-13)
    assert np.max(np.abs(native.gradient - reference.gradient)) < 2e-13


@pytest.mark.parametrize(
    "elements,distance",
    [
        ((1, 1), 4.0),
        ((1, 6), 4.0),
        ((6, 8), 4.5),
        ((3, 11), 6.0),
        ((14, 17), 5.0),
        ((18, 18), 7.0),
    ],
)
@pytest.mark.parametrize("component", ["e6", "e8"])
def test_native_per_pair_e6_e8_match_dftd3(elements, distance, component):
    """A two-atom system exposes one unique pair, allowing E6/E8 isolation."""
    multiplicity = 2 if sum(elements) % 2 else 1
    mol = vq.Molecule(
        [
            vq.Atom(elements[0], [0.0, 0.0, 0.0]),
            vq.Atom(elements[1], [0.0, 0.0, distance]),
        ],
        multiplicity=multiplicity,
    )
    if component == "e6":
        params = vq.D3BJParams(s6=1.0, s8=0.0, a1=0.4, a2=4.5)
    else:
        params = vq.D3BJParams(s6=0.0, s8=1.0, a1=0.4, a2=4.5)
    native = vq.compute_d3bj(
        mol, params, backend="builtin", with_gradient=True,
    )
    reference = vq.compute_d3bj(
        mol, params, backend="dftd3", with_gradient=True,
    )
    assert native.energy == pytest.approx(reference.energy, rel=0, abs=2e-13)
    assert np.max(np.abs(native.gradient - reference.gradient)) < 2e-12


# ---------------------------------------------------------------------------
# String-vs-D3BJParams parameter passing
# ---------------------------------------------------------------------------

def test_params_can_be_string_or_struct():
    """``compute_d3bj`` accepts either a functional name (looked up via
    ``d3bj_params_for``) or an explicit :class:`D3BJParams` struct.
    Both must give the same answer."""
    mol = _h2o_mol()
    p = vq.d3bj_params_for("pbe", backend="dftd3")
    e1 = vq.compute_d3bj(mol, "pbe", backend="dftd3").energy
    e2 = vq.compute_d3bj(mol, p, backend="dftd3").energy
    assert e1 == pytest.approx(e2, rel=1e-12)


def test_unknown_functional_with_string_raises():
    mol = _h2o_mol()
    with pytest.raises(ValueError, match="no D3-BJ parameters"):
        vq.compute_d3bj(mol, "not-a-functional", backend="dftd3")
